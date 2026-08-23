#include "wmfs/reference/kernels.hpp"

#include <ATen/ATen.h>
#include <wmfs/reference_plugin.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace wmfs::reference {
namespace {

struct ConfigurationState {
    std::string canonical;
    std::string precision;
    std::int64_t threads;
    bool emit_diagnostics;
};

std::mutex configuration_mutex;
std::unique_ptr<const ConfigurationState> configuration_state;

void set_error(wmfs_error_buffer_v1 *error, const char *message) {
    if (!error || error->struct_size < sizeof(*error) || !error->data ||
        !error->capacity)
        return;
    const std::size_t length = std::strlen(message);
    const std::size_t size =
        length < error->capacity ? length : error->capacity;
    std::memcpy(error->data, message, size);
    error->size = static_cast<std::uint32_t>(size);
    error->truncated = size != length;
}

std::string string_value(const std::string &json, const char *key,
                         const char *fallback) {
    const std::string prefix = std::string("\"") + key + "\":\"";
    const auto start = json.find(prefix);
    if (start == std::string::npos)
        return fallback;
    const auto value_start = start + prefix.size();
    const auto end = json.find('"', value_start);
    return end == std::string::npos
               ? std::string()
               : json.substr(value_start, end - value_start);
}

std::int64_t integer_value(const std::string &json, const char *key,
                           std::int64_t fallback) {
    const std::string prefix = std::string("\"") + key + "\":";
    const auto start = json.find(prefix);
    if (start == std::string::npos)
        return fallback;
    const char *first = json.c_str() + start + prefix.size();
    char *end = 0;
    const long long value = std::strtoll(first, &end, 10);
    return end == first ? fallback : static_cast<std::int64_t>(value);
}

bool boolean_value(const std::string &json, const char *key, bool fallback) {
    const std::string prefix = std::string("\"") + key + "\":";
    const auto start = json.find(prefix);
    if (start == std::string::npos)
        return fallback;
    return json.compare(start + prefix.size(), 4, "true") == 0;
}

at::Tensor tensor(const wmfs_tensor_v1 &value) {
    std::vector<std::int64_t> shape(value.shape, value.shape + value.rank);
    std::vector<std::int64_t> strides(value.strides,
                                      value.strides + value.rank);
    const auto dtype = static_cast<at::ScalarType>(
        value.dtype == WMFS_DTYPE_FLOAT32   ? at::kFloat
        : value.dtype == WMFS_DTYPE_FLOAT64 ? at::kDouble
        : value.dtype == WMFS_DTYPE_INT64   ? at::kLong
                                            : at::kByte);
    return at::from_blob(value.data, shape, strides,
                         at::TensorOptions().dtype(dtype).device(at::kCPU));
}

double floating_scalar(const wmfs_scalar_v1 &value) {
    double result;
    std::memcpy(&result, &value.bits, sizeof(result));
    return result;
}

std::int32_t invoke_matmul(const wmfs_invocation_v1 *value) {
    auto a = tensor(value->inputs[0]);
    auto b = tensor(value->inputs[1]);
    auto out = tensor(value->outputs[0]);
    wmfs::reference::matmul_out(a, b, out);
    return WMFS_STATUS_OK;
}

std::int32_t invoke_svd(const wmfs_invocation_v1 *value) {
    auto a = tensor(value->inputs[0]);
    auto u = tensor(value->outputs[0]);
    auto s = tensor(value->outputs[1]);
    auto vh = tensor(value->outputs[2]);
    wmfs::reference::svd_out(a, value->scalars[0].bits != 0, u, s, vh);
    return WMFS_STATUS_OK;
}

std::int32_t invoke_add_scalar(const wmfs_invocation_v1 *value) {
    auto a = tensor(value->inputs[0]);
    auto out = tensor(value->outputs[0]);
    wmfs::reference::add_scalar_out(a, floating_scalar(value->scalars[0]), out);
    return WMFS_STATUS_OK;
}

std::int32_t invoke_matmul_vjp(const wmfs_invocation_v1 *value) {
    auto a = tensor(value->inputs[0]);
    auto b = tensor(value->inputs[1]);
    auto cotangent = tensor(value->inputs[2]);
    auto a_gradient = tensor(value->outputs[0]);
    auto b_gradient = tensor(value->outputs[1]);
    wmfs::reference::matmul_vjp_out(a, b, cotangent, a_gradient, b_gradient);
    return WMFS_STATUS_OK;
}

std::int32_t invoke_add_scalar_vjp(const wmfs_invocation_v1 *value) {
    auto cotangent = tensor(value->inputs[0]);
    auto gradient = tensor(value->outputs[0]);
    wmfs::reference::add_scalar_vjp_out(cotangent, gradient);
    return WMFS_STATUS_OK;
}

std::int32_t invoke_nonzero(const wmfs_invocation_v1 *value) {
    auto a = tensor(value->inputs[0]);
    auto out = tensor(value->outputs[0]);
    wmfs::reference::nonzero_out(
        a, static_cast<std::int64_t>(value->scalars[0].bits), out);
    return WMFS_STATUS_OK;
}

std::int32_t plan_nonzero(const wmfs_invocation_v1 *value,
                          wmfs_output_plan_v1 *outputs, std::uint32_t capacity,
                          std::uint32_t *count) {
    if (capacity < 1)
        return WMFS_STATUS_INVALID_ARGUMENT;
    auto a = tensor(value->inputs[0]);
    const auto nonzero_count = at::count_nonzero(a).item<std::int64_t>();
    if (nonzero_count == 0)
        throw std::invalid_argument(
            "nonzero does not yet support an empty result");
    outputs[0] = {};
    outputs[0].struct_size = sizeof(outputs[0]);
    outputs[0].output_index = 0;
    outputs[0].dtype = WMFS_DTYPE_INT64;
    outputs[0].rank = 2;
    outputs[0].shape[0] = nonzero_count;
    outputs[0].shape[1] = a.dim();
    *count = 1;
    return WMFS_STATUS_OK;
}

} // namespace

std::int32_t initialize(wmfs_json_view_v1 configuration, logger log,
                        wmfs_error_buffer_v1 *error) {
    if ((!configuration.data && configuration.size) ||
        configuration.size > 65536) {
        set_error(error, "invalid configuration view");
        return WMFS_STATUS_INVALID_ARGUMENT;
    }
    const std::string canonical(configuration.data ? configuration.data : "",
                                static_cast<std::size_t>(configuration.size));
    ConfigurationState parsed{
        canonical, string_value(canonical, "precision", "balanced"),
        integer_value(canonical, "threads", 1),
        boolean_value(canonical, "emit_diagnostics", false)};
    if (canonical.empty() || canonical.front() != '{' ||
        canonical.back() != '}' || parsed.threads < 1 || parsed.threads > 64 ||
        (parsed.precision != "fast" && parsed.precision != "balanced" &&
         parsed.precision != "accurate")) {
        set_error(error, "reference configuration is semantically invalid");
        return WMFS_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(configuration_mutex);
    if (configuration_state) {
        if (configuration_state->canonical == canonical)
            return WMFS_STATUS_OK;
        set_error(error, "reference plugin is already initialized");
        return WMFS_STATUS_INVALID_ARGUMENT;
    }
    configuration_state.reset(new const ConfigurationState(parsed));
    if (parsed.emit_diagnostics && log.enabled(WMFS_LOG_INFO)) {
        static const char message[] = "reference plugin initialized";
        log.info(message, sizeof(message) - 1);
    }
    return WMFS_STATUS_OK;
}

void shutdown() {
    std::lock_guard<std::mutex> lock(configuration_mutex);
    configuration_state.reset();
}

#define WMFS_SPECIALIZE(operation, type, function)                             \
    template <>                                                                \
    std::int32_t operation##_typed<type>(dtype_tag<type>,                      \
                                         const wmfs_invocation_v1 *value) {    \
        return function(value);                                                \
    }
#define WMFS_NUMERIC(operation, function)                                      \
    WMFS_SPECIALIZE(operation, float, function)                                \
    WMFS_SPECIALIZE(operation, double, function)                               \
    WMFS_SPECIALIZE(operation, std::int64_t, function)                         \
    WMFS_SPECIALIZE(operation, std::uint8_t, function)
#define WMFS_PLAN_SPECIALIZE(type)                                             \
    template <>                                                                \
    std::int32_t nonzero_plan_typed<type>(                                     \
        dtype_tag<type>, const wmfs_invocation_v1 *value,                      \
        wmfs_output_plan_v1 *outputs, std::uint32_t capacity,                  \
        std::uint32_t *count) {                                                \
        return plan_nonzero(value, outputs, capacity, count);                  \
    }

WMFS_NUMERIC(matmul, invoke_matmul)
WMFS_SPECIALIZE(svd, float, invoke_svd)
WMFS_SPECIALIZE(svd, double, invoke_svd)
WMFS_NUMERIC(add_scalar, invoke_add_scalar)
WMFS_NUMERIC(matmul_vjp, invoke_matmul_vjp)
WMFS_NUMERIC(add_scalar_vjp, invoke_add_scalar_vjp)
WMFS_NUMERIC(nonzero, invoke_nonzero)

WMFS_PLAN_SPECIALIZE(float)
WMFS_PLAN_SPECIALIZE(double)
WMFS_PLAN_SPECIALIZE(std::int64_t)
WMFS_PLAN_SPECIALIZE(std::uint8_t)

#undef WMFS_NUMERIC
#undef WMFS_PLAN_SPECIALIZE
#undef WMFS_SPECIALIZE

} // namespace wmfs::reference
