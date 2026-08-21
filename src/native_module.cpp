#include "wmfs/native/session.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using wmfs::native::InvocationOutcome;
using wmfs::native::InvocationProfile;
using wmfs::native::Mapping;
using wmfs::native::ScalarArgument;
using wmfs::native::ScalarKind;
using wmfs::native::Session;
using wmfs::native::TensorDescriptor;
using wmfs::native::TensorDescriptors;
using wmfs::native::TensorDType;

namespace {

std::uint64_t integer(nb::handle value) {
    return nb::cast<std::uint64_t>(value);
}

TensorDType dtype_from_object(nb::handle value) {
    const auto name = nb::cast<std::string>(value);
    if (name == "float32")
        return TensorDType::float32;
    if (name == "float64")
        return TensorDType::float64;
    if (name == "int64")
        return TensorDType::int64;
    if (name == "uint8")
        return TensorDType::uint8;
    throw std::invalid_argument("Unsupported tensor dtype: " + name);
}

TensorDescriptor descriptor_from_object(nb::handle value) {
    return TensorDescriptor{
        .buffer_id = integer(value.attr("buffer_id")),
        .generation = nb::cast<std::uint32_t>(value.attr("generation")),
        .allocation_id = integer(value.attr("allocation_id")),
        .offset = integer(value.attr("offset")),
        .byte_length = integer(value.attr("byte_length")),
        .dtype = dtype_from_object(value.attr("dtype")),
        .shape = nb::cast<std::vector<std::uint64_t>>(value.attr("shape")),
        .strides = nb::cast<std::vector<std::int64_t>>(value.attr("strides")),
    };
}

struct DescriptorReferences {
    TensorDescriptors values;
    std::vector<nb::object> owners;
};

DescriptorReferences descriptors_from_list(const nb::list &values) {
    DescriptorReferences result;
    result.values.reserve(values.size());
    result.owners.reserve(values.size());
    for (nb::handle value : values) {
        result.owners.push_back(nb::borrow<nb::object>(value));
        result.values.push_back(&nb::cast<const TensorDescriptor &>(value));
    }
    return result;
}

ScalarArgument scalar_from_tuple(const nb::tuple &value) {
    ScalarArgument result{};
    result.parameter = nb::cast<std::uint16_t>(value[0]);
    const auto kind = nb::cast<std::string>(value[1]);
    if (kind == "boolean") {
        result.kind = ScalarKind::boolean;
        result.boolean_value = nb::cast<bool>(value[2]);
    } else if (kind == "float64") {
        result.kind = ScalarKind::float64;
        result.float64_value = nb::cast<double>(value[2]);
    } else if (kind == "int64") {
        result.kind = ScalarKind::int64;
        result.int64_value = nb::cast<std::int64_t>(value[2]);
    } else if (kind == "text") {
        result.kind = ScalarKind::text;
        result.text_value = nb::cast<std::string>(value[2]);
    } else {
        throw std::invalid_argument("Unknown scalar kind: " + kind);
    }
    return result;
}

std::vector<ScalarArgument> scalars_from_list(const nb::list &values) {
    std::vector<ScalarArgument> result;
    result.reserve(values.size());
    for (nb::handle value : values) {
        result.push_back(scalar_from_tuple(nb::cast<nb::tuple>(value)));
    }
    return result;
}

Mapping mapping_from_buffer(nb::handle buffer, std::uint64_t invocation_id,
                            bool writable) {
    const bool arena = nb::cast<bool>(buffer.attr("arena"));
    return Mapping{
        .buffer_id = nb::cast<std::uint64_t>(buffer.attr("id")),
        .generation = nb::cast<std::uint32_t>(buffer.attr("generation")),
        .allocation_id = nb::cast<std::uint64_t>(buffer.attr("allocation_id")),
        .byte_length =
            nb::cast<std::uint64_t>(buffer.attr("mapping_byte_length")),
        .writable = writable || arena,
        .arena = arena,
        .invocation_id = invocation_id,
    };
}

bool ensure_mapped(Session &session, nb::object buffer,
                   std::uint64_t invocation_id, bool writable) {
    const auto mapping = mapping_from_buffer(buffer, invocation_id, writable);
    if (!session.mapping_required(mapping))
        return false;
    const int fd = nb::cast<int>(buffer.attr("duplicate_fd")(mapping.writable));
    nb::gil_scoped_release release;
    session.map_buffer(mapping, fd);
    return true;
}

nb::list ensure_mapped_many(Session &session, const nb::list &buffers,
                            std::uint64_t invocation_id) {
    std::vector<std::pair<Mapping, int>> pending;
    std::vector<std::size_t> pending_indices;
    std::vector<bool> results(buffers.size(), false);
    try {
        for (std::size_t index = 0; index < buffers.size(); ++index) {
            auto item = nb::cast<nb::tuple>(buffers[index]);
            auto buffer = nb::borrow<nb::object>(item[0]);
            auto mapping = mapping_from_buffer(buffer, invocation_id,
                                               nb::cast<bool>(item[1]));
            if (!session.mapping_required(mapping))
                continue;
            int fd =
                nb::cast<int>(buffer.attr("duplicate_fd")(mapping.writable));
            pending.emplace_back(mapping, fd);
            pending_indices.push_back(index);
        }
    } catch (...) {
        for (const auto &item : pending)
            ::close(item.second);
        throw;
    }
    if (!pending.empty()) {
        std::vector<bool> mapped;
        {
            nb::gil_scoped_release release;
            mapped = session.map_buffers(std::move(pending));
        }
        for (std::size_t index = 0; index < mapped.size(); ++index)
            results[pending_indices[index]] = mapped[index];
    }
    nb::list result;
    for (bool mapped : results)
        result.append(mapped);
    return result;
}

void retire_buffer(Session &session, nb::object buffer) {
    const auto mapping = mapping_from_buffer(buffer, 0, false);
    nb::gil_scoped_release release;
    session.retire_buffer(mapping);
}

void retire_buffers(Session &session, const nb::list &buffers) {
    std::vector<Mapping> mappings;
    mappings.reserve(buffers.size());
    for (nb::handle buffer : buffers)
        mappings.push_back(mapping_from_buffer(buffer, 0, false));
    nb::gil_scoped_release release;
    session.retire_buffers(mappings);
}

nb::dict outcome_dict(const InvocationOutcome &outcome) {
    nb::dict result;
    result["error_type"] = outcome.error_type;
    result["error_message"] = outcome.error_message;
    return result;
}

nb::dict invoke(Session &session, std::uint64_t invocation_id,
                std::uint32_t operation_id, const nb::list &inputs,
                const nb::list &outputs, const nb::list &scalars) {
    auto native_inputs = descriptors_from_list(inputs);
    auto native_outputs = descriptors_from_list(outputs);
    auto native_scalars = scalars_from_list(scalars);
    InvocationOutcome outcome;
    {
        nb::gil_scoped_release release;
        outcome =
            session.invoke(invocation_id, operation_id, native_inputs.values,
                           native_outputs.values, native_scalars);
    }
    return outcome_dict(outcome);
}

nb::dict invoke_profiled(Session &session, std::uint64_t invocation_id,
                         std::uint32_t operation_id, const nb::list &inputs,
                         const nb::list &outputs, const nb::list &scalars) {
    auto native_inputs = descriptors_from_list(inputs);
    auto native_outputs = descriptors_from_list(outputs);
    auto native_scalars = scalars_from_list(scalars);
    InvocationProfile profile;
    {
        nb::gil_scoped_release release;
        profile = session.invoke_profiled(
            invocation_id, operation_id, native_inputs.values,
            native_outputs.values, native_scalars);
    }
    nb::dict result;
    result["error_type"] = profile.outcome.error_type;
    result["error_message"] = profile.outcome.error_message;
    result["queue_wait_ns"] = profile.queue_wait_ns;
    result["rpc_ns"] = profile.rpc_ns;
    result["worker_input_views_ns"] = profile.worker_input_views_ns;
    result["worker_output_views_ns"] = profile.worker_output_views_ns;
    result["worker_dispatch_ns"] = profile.worker_dispatch_ns;
    result["worker_kernel_ns"] = profile.worker_kernel_ns;
    return result;
}

nb::bytes metadata(Session &session) {
    auto value = session.metadata();
    return nb::bytes(reinterpret_cast<const char *>(value.data()),
                     value.size());
}

nb::bytes environment(Session &session) {
    auto value = session.environment();
    return nb::bytes(reinterpret_cast<const char *>(value.data()),
                     value.size());
}

} // namespace

NB_MODULE(_native, module) {
    nb::class_<TensorDescriptor>(module, "_TensorDescriptor");
    module.def("_make_tensor_descriptor", &descriptor_from_object,
               "descriptor"_a);
    nb::class_<Session>(module, "Session")
        .def(nb::init<int, int, std::uint64_t, double, double, double>(),
             "rpc_fd"_a, "control_fd"_a, "expected_fingerprint"_a,
             "startup_timeout_seconds"_a, "request_timeout_seconds"_a,
             "fd_transfer_timeout_seconds"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("ensure_mapped", &ensure_mapped, "buffer"_a, "invocation_id"_a,
             "writable"_a = false)
        .def("ensure_mapped_many", &ensure_mapped_many, "buffers"_a,
             "invocation_id"_a)
        .def("retire_buffer", &retire_buffer, "buffer"_a)
        .def("retire_buffers", &retire_buffers, "buffers"_a)
        .def("abort_invocation", &Session::abort_invocation, "invocation_id"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def("invoke", &invoke, "invocation_id"_a, "operation_id"_a, "inputs"_a,
             "outputs"_a, "scalars"_a)
        .def("invoke_profiled", &invoke_profiled, "invocation_id"_a,
             "operation_id"_a, "inputs"_a, "outputs"_a, "scalars"_a)
        .def("ping", &Session::ping, "nonce"_a,
             nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("metadata", &metadata)
        .def("environment", &environment)
        .def("close", &Session::close, nb::call_guard<nb::gil_scoped_release>())
        .def_prop_ro("transfer_count", &Session::transfer_count)
        .def_prop_ro("mapping_batch_count", &Session::mapping_batch_count)
        .def_prop_ro("retirement_count", &Session::retirement_count)
        .def_prop_ro("retirement_batch_count",
                     &Session::retirement_batch_count);
}
