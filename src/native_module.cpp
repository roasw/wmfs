#include "wmfs/native/session.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using wmfs::native::InvocationOutcome;
using wmfs::native::InvocationProfile;
using wmfs::native::Mapping;
using wmfs::native::OutputPlanningResult;
using wmfs::native::RingSubmissionResult;
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

const char *dtype_name(TensorDType dtype) {
    switch (dtype) {
    case TensorDType::float32:
        return "float32";
    case TensorDType::float64:
        return "float64";
    case TensorDType::int64:
        return "int64";
    case TensorDType::uint8:
        return "uint8";
    }
    throw std::invalid_argument("Unsupported tensor dtype");
}

std::uint32_t ring_dtype_from_object(nb::handle value) {
    const auto name = nb::cast<std::string>(value);
    if (name == "bool")
        return WMFS_RING_DTYPE_BOOL;
    if (name == "int8")
        return WMFS_RING_DTYPE_INT8;
    if (name == "uint8")
        return WMFS_RING_DTYPE_UINT8;
    if (name == "int16")
        return WMFS_RING_DTYPE_INT16;
    if (name == "int32")
        return WMFS_RING_DTYPE_INT32;
    if (name == "int64")
        return WMFS_RING_DTYPE_INT64;
    if (name == "float16")
        return WMFS_RING_DTYPE_FLOAT16;
    if (name == "float32")
        return WMFS_RING_DTYPE_FLOAT32;
    if (name == "float64")
        return WMFS_RING_DTYPE_FLOAT64;
    if (name == "bfloat16")
        return WMFS_RING_DTYPE_BFLOAT16;
    throw std::invalid_argument("Unsupported ring dtype: " + name);
}

const char *ring_dtype_name(std::uint32_t dtype) {
    switch (dtype) {
    case WMFS_RING_DTYPE_BOOL:
        return "bool";
    case WMFS_RING_DTYPE_INT8:
        return "int8";
    case WMFS_RING_DTYPE_UINT8:
        return "uint8";
    case WMFS_RING_DTYPE_INT16:
        return "int16";
    case WMFS_RING_DTYPE_INT32:
        return "int32";
    case WMFS_RING_DTYPE_INT64:
        return "int64";
    case WMFS_RING_DTYPE_FLOAT16:
        return "float16";
    case WMFS_RING_DTYPE_FLOAT32:
        return "float32";
    case WMFS_RING_DTYPE_FLOAT64:
        return "float64";
    case WMFS_RING_DTYPE_BFLOAT16:
        return "bfloat16";
    default:
        throw std::invalid_argument("Unsupported ring dtype ID");
    }
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

wmfs_ring_record_v1 ring_record_from_object(nb::handle value) {
    wmfs_ring_record_v1 record{};
    record.kind = nb::cast<std::uint32_t>(value.attr("kind"));
    record.flags = nb::cast<std::uint32_t>(value.attr("flags"));
    record.session_generation =
        nb::cast<std::uint64_t>(value.attr("generation"));
    record.invocation_id = nb::cast<std::uint64_t>(value.attr("invocation_id"));
    record.operation_id = nb::cast<std::uint32_t>(value.attr("operation_id"));
    record.status = nb::cast<std::uint32_t>(value.attr("status"));

    const auto tensors = nb::cast<nb::tuple>(value.attr("tensors"));
    const auto scalars = nb::cast<nb::tuple>(value.attr("scalars"));
    const auto outputs = nb::cast<nb::tuple>(value.attr("outputs"));
    if (tensors.size() > 16 || scalars.size() > 16 || outputs.size() > 16)
        throw std::invalid_argument("Ring descriptor count exceeds ABI bound");
    record.tensor_count = static_cast<std::uint16_t>(tensors.size());
    record.scalar_count = static_cast<std::uint16_t>(scalars.size());
    record.planned_output_count = static_cast<std::uint16_t>(outputs.size());

    for (std::size_t index = 0; index < tensors.size(); ++index) {
        auto item = tensors[index];
        auto &tensor = record.tensors[index];
        tensor.buffer_id = nb::cast<std::uint64_t>(item.attr("buffer_id"));
        tensor.buffer_generation =
            nb::cast<std::uint64_t>(item.attr("generation"));
        tensor.allocation_id =
            nb::cast<std::uint64_t>(item.attr("allocation_id"));
        tensor.byte_offset = nb::cast<std::uint64_t>(item.attr("offset"));
        tensor.byte_length = nb::cast<std::uint64_t>(item.attr("byte_length"));
        tensor.dtype = ring_dtype_from_object(item.attr("dtype"));
        tensor.kind = nb::cast<std::uint16_t>(item.attr("kind"));
        tensor.parameter_index =
            nb::cast<std::uint16_t>(item.attr("parameter"));
        tensor.flags = nb::cast<bool>(item.attr("writable"))
                           ? WMFS_RING_TENSOR_FLAG_WRITABLE
                           : 0;
        const auto shape =
            nb::cast<std::vector<std::int64_t>>(item.attr("shape"));
        const auto strides =
            nb::cast<std::vector<std::int64_t>>(item.attr("strides"));
        if (shape.empty() || shape.size() > 16 ||
            shape.size() != strides.size())
            throw std::invalid_argument("Ring tensor rank is invalid");
        tensor.rank = static_cast<std::uint16_t>(shape.size());
        std::copy(shape.begin(), shape.end(), tensor.shape);
        std::copy(strides.begin(), strides.end(), tensor.strides);
    }

    for (std::size_t index = 0; index < scalars.size(); ++index) {
        auto item = scalars[index];
        auto &scalar = record.scalars[index];
        scalar.parameter_index =
            nb::cast<std::uint16_t>(item.attr("parameter"));
        const auto kind = nb::cast<std::string>(item.attr("kind"));
        auto scalar_value = item.attr("value");
        if (kind == "boolean") {
            scalar.kind = WMFS_RING_SCALAR_BOOLEAN;
            scalar.bits = nb::cast<bool>(scalar_value);
        } else if (kind == "float64") {
            scalar.kind = WMFS_RING_SCALAR_FLOAT64;
            const auto number = nb::cast<double>(scalar_value);
            std::memcpy(&scalar.bits, &number, sizeof(number));
        } else if (kind == "int64") {
            scalar.kind = WMFS_RING_SCALAR_INT64;
            const auto number = nb::cast<std::int64_t>(scalar_value);
            std::memcpy(&scalar.bits, &number, sizeof(number));
        } else if (kind == "text") {
            scalar.kind = WMFS_RING_SCALAR_TEXT;
            const auto text = nb::cast<std::string>(scalar_value);
            if (text.size() > sizeof(scalar.text))
                throw std::invalid_argument(
                    "Scalar text exceeds ring ABI bound");
            scalar.text_length = static_cast<std::uint32_t>(text.size());
            std::memcpy(scalar.text, text.data(), text.size());
        } else {
            throw std::invalid_argument("Unknown ring scalar kind: " + kind);
        }
    }

    for (std::size_t index = 0; index < outputs.size(); ++index) {
        auto item = outputs[index];
        auto &output = record.planned_outputs[index];
        output.dtype = ring_dtype_from_object(item.attr("dtype"));
        output.output_index = nb::cast<std::uint16_t>(item.attr("output"));
        const auto shape =
            nb::cast<std::vector<std::int64_t>>(item.attr("shape"));
        if (shape.empty() || shape.size() > 16)
            throw std::invalid_argument("Planned output rank is invalid");
        output.rank = static_cast<std::uint16_t>(shape.size());
        std::copy(shape.begin(), shape.end(), output.shape);
    }
    return record;
}

nb::dict ring_completion_dict(const wmfs_ring_record_v1 &record) {
    nb::dict result;
    result["kind"] = record.kind;
    result["flags"] = record.flags;
    result["generation"] = record.session_generation;
    result["submission_id"] = record.submission_id;
    result["invocation_id"] = record.invocation_id;
    result["operation_id"] = record.operation_id;
    result["status"] = record.status;
    nb::list outputs;
    for (std::size_t index = 0; index < record.planned_output_count; ++index) {
        const auto &output = record.planned_outputs[index];
        nb::list shape;
        for (std::size_t axis = 0; axis < output.rank; ++axis)
            shape.append(output.shape[axis]);
        outputs.append(nb::make_tuple(output.output_index, nb::tuple(shape),
                                      ring_dtype_name(output.dtype)));
    }
    result["outputs"] = outputs;
    result["profile"] = nb::make_tuple(
        record.profile.command_published_ns, record.profile.worker_dequeued_ns,
        record.profile.worker_started_ns, record.profile.worker_input_views_ns,
        record.profile.worker_output_views_ns,
        record.profile.worker_dispatch_ns, record.profile.worker_kernel_ns,
        record.profile.completion_published_ns);
    result["error_type"] =
        std::string(record.error.type, record.error.type_length);
    result["error_message"] =
        std::string(record.error.message, record.error.message_length);
    return result;
}

nb::dict ring_metrics_dict(const RingSubmissionResult &result) {
    const auto &metrics = result.metrics;
    nb::dict values;
    values["round_trip_ns"] = metrics.round_trip_ns;
    values["submission_queue_ns"] = metrics.submission_queue_ns;
    values["enqueue_ns"] = metrics.enqueue_ns;
    values["backpressure_wait_ns"] = metrics.backpressure_wait_ns;
    values["command_wakeup_ns"] = metrics.command_wakeup_ns;
    values["worker_queue_ns"] = metrics.worker_queue_ns;
    values["worker_kernel_ns"] = metrics.worker_kernel_ns;
    values["completion_wakeup_ns"] = metrics.completion_wakeup_ns;
    values["result_materialization_ns"] = metrics.result_materialization_ns;
    return values;
}

nb::tuple submit_ring(Session &session, nb::object record, double timeout,
                      bool profiled) {
    auto command = ring_record_from_object(record);
    RingSubmissionResult result;
    {
        nb::gil_scoped_release release;
        result = session.submit_ring(command, timeout, profiled);
    }
    return nb::make_tuple(ring_completion_dict(result.completion),
                          ring_metrics_dict(result));
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

nb::dict plan_outputs(Session &session, std::uint64_t invocation_id,
                      std::uint32_t operation_id, const nb::list &inputs,
                      const nb::list &scalars) {
    auto native_inputs = descriptors_from_list(inputs);
    auto native_scalars = scalars_from_list(scalars);
    OutputPlanningResult planned;
    {
        nb::gil_scoped_release release;
        planned = session.plan_outputs(invocation_id, operation_id,
                                       native_inputs.values, native_scalars);
    }
    nb::dict result = outcome_dict(planned.outcome);
    nb::list outputs;
    for (const auto &item : planned.outputs)
        outputs.append(
            nb::make_tuple(item.output, item.shape, dtype_name(item.dtype)));
    result["outputs"] = outputs;
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
        .def(nb::init<int, int, std::uint64_t, double, double, double,
                      std::uint64_t, std::uint32_t, int, int, int, int, int,
                      int>(),
             "rpc_fd"_a, "control_fd"_a, "expected_fingerprint"_a,
             "startup_timeout_seconds"_a, "request_timeout_seconds"_a,
             "fd_transfer_timeout_seconds"_a, "ring_generation"_a = 0,
             "ring_capacity"_a = 0, "command_ring_fd"_a = -1,
             "command_data_fd"_a = -1, "command_space_fd"_a = -1,
             "completion_ring_fd"_a = -1, "completion_data_fd"_a = -1,
             "completion_space_fd"_a = -1,
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
        .def("plan_outputs", &plan_outputs, "invocation_id"_a, "operation_id"_a,
             "inputs"_a, "scalars"_a)
        .def("submit_ring", &submit_ring, "record"_a, "timeout"_a,
             "profiled"_a = false)
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
