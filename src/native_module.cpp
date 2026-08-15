#include "wmfs/native/session.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <unistd.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using wmfs::native::InvocationProfile;
using wmfs::native::Mapping;
using wmfs::native::ScalarArgument;
using wmfs::native::ScalarKind;
using wmfs::native::Session;
using wmfs::native::TensorDescriptor;

namespace {

std::uint64_t integer(nb::handle value) {
    return nb::cast<std::uint64_t>(value);
}

TensorDescriptor descriptor_from_object(nb::handle value) {
    return TensorDescriptor{
        .buffer_id = integer(value.attr("buffer_id")),
        .generation = nb::cast<std::uint32_t>(value.attr("generation")),
        .allocation_id = integer(value.attr("allocation_id")),
        .offset = integer(value.attr("offset")),
        .byte_length = integer(value.attr("byte_length")),
        .dtype = nb::cast<std::string>(value.attr("dtype")),
        .shape = nb::cast<std::vector<std::uint64_t>>(value.attr("shape")),
        .strides = nb::cast<std::vector<std::int64_t>>(value.attr("strides")),
    };
}

std::vector<TensorDescriptor> descriptors_from_list(const nb::list &values) {
    std::vector<TensorDescriptor> result;
    result.reserve(values.size());
    for (nb::handle value : values) {
        result.push_back(descriptor_from_object(value));
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

void retire_buffer(Session &session, nb::object buffer) {
    const auto mapping = mapping_from_buffer(buffer, 0, false);
    nb::gil_scoped_release release;
    session.retire_buffer(mapping);
}

nb::object invoke(Session &session, std::uint64_t invocation_id,
                  std::uint32_t operation_id, const nb::list &inputs,
                  const nb::list &outputs, const nb::list &scalars,
                  bool profiled) {
    auto native_inputs = descriptors_from_list(inputs);
    auto native_outputs = descriptors_from_list(outputs);
    auto native_scalars = scalars_from_list(scalars);
    InvocationProfile profile;
    {
        nb::gil_scoped_release release;
        profile = session.invoke(invocation_id, operation_id, native_inputs,
                                 native_outputs, native_scalars, profiled);
    }
    if (!profiled)
        return nb::none();
    nb::dict result;
    result["queue_wait_ns"] = profile.queue_wait_ns;
    result["rpc_ns"] = profile.rpc_ns;
    result["worker_input_views_ns"] = profile.worker_input_views_ns;
    result["worker_output_views_ns"] = profile.worker_output_views_ns;
    result["worker_dispatch_ns"] = profile.worker_dispatch_ns;
    result["worker_kernel_ns"] = profile.worker_kernel_ns;
    return result;
}

} // namespace

NB_MODULE(_native, module) {
    nb::class_<Session>(module, "Session")
        .def(nb::init<int, int, std::uint64_t>(), "rpc_fd"_a, "control_fd"_a,
             "expected_fingerprint"_a, nb::call_guard<nb::gil_scoped_release>())
        .def("ensure_mapped", &ensure_mapped, "buffer"_a, "invocation_id"_a,
             "writable"_a = false)
        .def("retire_buffer", &retire_buffer, "buffer"_a)
        .def("invoke", &invoke, "invocation_id"_a, "operation_id"_a, "inputs"_a,
             "outputs"_a, "scalars"_a, "profiled"_a = false)
        .def(
            "ping",
            [](Session &session, std::uint64_t nonce) {
                nb::gil_scoped_release release;
                session.ping(nonce);
            },
            "nonce"_a)
        .def("close",
             [](Session &session) {
                 nb::gil_scoped_release release;
                 session.close();
             })
        .def_prop_ro("transfer_count", &Session::transfer_count)
        .def_prop_ro("retirement_count", &Session::retirement_count);
}
