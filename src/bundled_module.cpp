#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <wmfs/bundled_registry.hpp>

#include <cstddef>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

namespace nb = nanobind;

namespace {
struct Instance {
    const wmfs_plugin_api_v1 *api;
    std::string configuration;
    std::size_t references;
    nb::object logger;
};

std::mutex instances_mutex;
std::map<std::string, Instance> instances;

bool has_lifecycle_fields(const wmfs_plugin_api_v1 *api) {
    return api && api->struct_size >= offsetof(wmfs_plugin_api_v1, shutdown) +
                                          sizeof(api->shutdown);
}

std::uint8_t log_enabled(void *context, std::uint32_t level) {
    try {
        nb::gil_scoped_acquire acquire;
        auto *instance = static_cast<Instance *>(context);
        return instance && !instance->logger.is_none() &&
               nb::cast<bool>(instance->logger.attr("enabled")(level));
    } catch (...) {
        return 0;
    }
}

void log_write(void *context, std::uint32_t level, wmfs_text_view_v1 message,
               wmfs_text_view_v1 category, const wmfs_log_field_v1 *fields,
               std::uint32_t field_count) {
    try {
        nb::gil_scoped_acquire acquire;
        auto *instance = static_cast<Instance *>(context);
        if (!instance || instance->logger.is_none())
            return;
        nb::dict values;
        for (std::uint32_t index = 0; index < field_count; ++index) {
            const auto &field = fields[index];
            const nb::str name(field.name.data ? field.name.data : "",
                               field.name.size);
            if (field.kind == WMFS_LOG_FIELD_BOOLEAN)
                values[name] = nb::bool_(field.bits != 0);
            else if (field.kind == WMFS_LOG_FIELD_INT64)
                values[name] = nb::int_(static_cast<std::int64_t>(field.bits));
            else if (field.kind == WMFS_LOG_FIELD_UINT64)
                values[name] = nb::int_(field.bits);
            else if (field.kind == WMFS_LOG_FIELD_FLOAT64) {
                double value;
                std::memcpy(&value, &field.bits, sizeof(value));
                values[name] = nb::float_(value);
            } else if (field.kind == WMFS_LOG_FIELD_TEXT)
                values[name] = nb::str(field.text.data ? field.text.data : "",
                                       field.text.size);
        }
        instance->logger.attr("log")(
            level, nb::str(message.data ? message.data : "", message.size),
            nb::str(category.data ? category.data : "", category.size), values);
    } catch (...) {
    }
}

void initialize_plugin(const std::string &name, nb::bytes encoded,
                       nb::object logger) {
    const std::string configuration(encoded.c_str(), encoded.size());
    std::lock_guard<std::mutex> lock(instances_mutex);
    const auto existing = instances.find(name);
    if (existing != instances.end()) {
        if (existing->second.configuration != configuration)
            throw std::runtime_error("bundled plugin is already initialized "
                                     "with different configuration");
        ++existing->second.references;
        return;
    }
    const auto *api = wmfs_bundled_plugin_api(name.c_str());
    if (!api)
        throw std::runtime_error("bundled plugin API is unavailable");
    auto inserted = instances.emplace(
        name, Instance{api, configuration, 1, std::move(logger)});
    auto &instance = inserted.first->second;
    if (has_lifecycle_fields(api) &&
        (api->features & WMFS_PLUGIN_FEATURE_INITIALIZE)) {
        if (!api->initialize) {
            instances.erase(inserted.first);
            throw std::runtime_error(
                "bundled plugin initialize callback is missing");
        }
        char error_data[1024] = {};
        wmfs_error_buffer_v1 error = {sizeof(error), sizeof(error_data),
                                      error_data, 0, 0};
        wmfs_initialize_args_v1 args = {};
        args.struct_size = sizeof(args);
        args.features = api->features;
        args.configuration.data = configuration.data();
        args.configuration.size = configuration.size();
        args.logger.struct_size = sizeof(args.logger);
        if (!instance.logger.is_none()) {
            args.logger.context = &instance;
            args.logger.enabled = &log_enabled;
            args.logger.log = &log_write;
        }
        args.error = &error;
        const auto status = api->initialize(&args);
        if (status != WMFS_STATUS_OK) {
            const std::size_t size =
                error.size < error.capacity ? error.size : error.capacity;
            instances.erase(inserted.first);
            throw std::runtime_error(
                size ? std::string(error.data, size)
                     : "bundled plugin rejected initialization");
        }
    }
}

void shutdown_plugin(const std::string &name) {
    std::lock_guard<std::mutex> lock(instances_mutex);
    const auto existing = instances.find(name);
    if (existing == instances.end())
        return;
    if (--existing->second.references)
        return;
    const auto *api = existing->second.api;
    if (has_lifecycle_fields(api) &&
        (api->features & WMFS_PLUGIN_FEATURE_SHUTDOWN) && api->shutdown)
        api->shutdown();
    instances.erase(existing);
}
} // namespace

NB_MODULE(_bundled, module) {
    nb::list plugins;
    std::istringstream names(WMFS_BUNDLED_PLUGIN_NAMES);
    for (std::string name; std::getline(names, name, ',');) {
        if (!name.empty())
            plugins.append(nb::str(name.c_str()));
    }
    module.attr("plugins") = nb::tuple(plugins);
    module.def("initialize", &initialize_plugin, nb::arg("plugin"),
               nb::arg("configuration"), nb::arg("logger") = nb::none());
    module.def("shutdown", &shutdown_plugin, nb::arg("plugin"));
}
