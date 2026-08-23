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
};

std::mutex instances_mutex;
std::map<std::string, Instance> instances;

bool has_lifecycle_fields(const wmfs_plugin_api_v1 *api) {
    return api && api->struct_size >= offsetof(wmfs_plugin_api_v1, shutdown) +
                                          sizeof(api->shutdown);
}

void initialize_plugin(const std::string &name, nb::bytes encoded) {
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
    if (has_lifecycle_fields(api) &&
        (api->features & WMFS_PLUGIN_FEATURE_INITIALIZE)) {
        if (!api->initialize)
            throw std::runtime_error(
                "bundled plugin initialize callback is missing");
        char error_data[1024] = {};
        wmfs_error_buffer_v1 error = {sizeof(error), sizeof(error_data),
                                      error_data, 0, 0};
        wmfs_initialize_args_v1 args = {};
        args.struct_size = sizeof(args);
        args.features = api->features;
        args.configuration.data = configuration.data();
        args.configuration.size = configuration.size();
        args.logger.struct_size = sizeof(args.logger);
        args.error = &error;
        const auto status = api->initialize(&args);
        if (status != WMFS_STATUS_OK) {
            const std::size_t size =
                error.size < error.capacity ? error.size : error.capacity;
            throw std::runtime_error(
                size ? std::string(error.data, size)
                     : "bundled plugin rejected initialization");
        }
    }
    instances.emplace(name, Instance{api, configuration, 1});
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
               nb::arg("configuration"));
    module.def("shutdown", &shutdown_plugin, nb::arg("plugin"));
}
