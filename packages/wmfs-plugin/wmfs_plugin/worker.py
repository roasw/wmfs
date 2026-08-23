import argparse
import ctypes
import json
import socket
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter_ns, sleep
from typing import TypeAlias

import torch

from wmfs_plugin.control import (
    DescriptorRole,
    ErrorResponse,
    Kind,
    Startup,
    Status,
    close_fds,
    decode_frame,
    decode_startup,
    encode_empty,
    encode_error,
    encode_startup,
    recvmsg_strict,
    sendmsg_strict,
)
from wmfs_plugin.fd_transport import FdReceiver, MappedBufferCache
from wmfs_plugin.invocation import InvocationContext, OutputSpec
from wmfs_plugin.metadata import (
    DimensionExpression,
    DTypeExpression,
    DTypeVariable,
    InputAxis,
    KnownOutput,
    OperationMetadata,
    OutputPlan,
    PromoteTensorScalar,
    ScalarParameter,
    SelectDimension,
    TensorParameter,
    VjpMetadata,
)
from wmfs_plugin.ring import (
    COMMAND_INVOKE,
    COMMAND_PING,
    COMMAND_PLAN_OUTPUTS,
    COMPLETION_INVOKE,
    COMPLETION_PLAN_OUTPUTS,
    COMPLETION_PONG,
    FLAG_PROFILE,
    STATUS_INTERNAL_ERROR,
    STATUS_OK,
    STATUS_OPERATION_ERROR,
    PlannedOutput,
    Record,
    RingEndpoint,
    RingError,
)

OperationHandler: TypeAlias = Callable[[InvocationContext], None]
OutputPlanner: TypeAlias = Callable[[InvocationContext], Mapping[str, OutputSpec]]


@dataclass(frozen=True)
class _Operation:
    handler: OperationHandler
    metadata: OperationMetadata
    input_accesses: tuple[str, ...]
    scalar_kinds: tuple[str, ...]


def worker_main(
    operations: Mapping[str, OperationHandler],
    *,
    output_planners: Mapping[str, OutputPlanner] | None = None,
) -> None:
    """Run a metadata-driven plugin worker on inherited transport descriptors.

    Args:
        operations: Mapping from every declared operation name, including
            internal VJP operations, to an invocation-context handler.

    Note:
        The worker entry point is launched by WMFS with only a bootstrap socket.
        Generated declarations attached by ``bind_operations`` provide identity.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-fd", type=int, required=True)
    arguments = parser.parse_args()
    _serve(
        arguments.bootstrap_fd,
        operations,
        output_planners or {},
    )


def _serve(
    bootstrap_fd: int,
    operations: Mapping[str, OperationHandler],
    output_planners: Mapping[str, OutputPlanner],
) -> None:
    bootstrap = socket.socket(fileno=bootstrap_fd)
    packet, descriptors = recvmsg_strict(bootstrap)
    request_id = 0
    try:
        request_id, startup = decode_startup(packet)
        expected_roles = (
            DescriptorRole.COMMAND_RING,
            DescriptorRole.COMMAND_DATA_EVENT,
            DescriptorRole.COMMAND_SPACE_EVENT,
            DescriptorRole.COMPLETION_RING,
            DescriptorRole.COMPLETION_DATA_EVENT,
            DescriptorRole.COMPLETION_SPACE_EVENT,
            DescriptorRole.FD_CONTROL,
        )
        if startup.descriptor_roles != expected_roles or len(descriptors) != 7:
            raise ValueError("STARTUP_REQUEST descriptor roles are not exact")
        _validate_startup_identity(startup, operations)
        config = json.loads(startup.config)
        if not isinstance(config, dict) or _canonical_json(config) != startup.config:
            raise ValueError("startup configuration is not a canonical JSON object")
        command_fds = tuple(descriptors[:3])
        completion_fds = tuple(descriptors[3:6])
        fd_socket_fd = descriptors[6]
        descriptors.clear()
        environment = _canonical_json(
            {
                "executable": sys.executable,
                "glibcVersion": _glibc_version(),
                "pythonVersion": sys.version.split()[0],
                "torchVersion": torch.__version__,
                "configuration": config,
            }
        )
        response = Startup(
            startup.session_generation,
            startup.interface_fingerprint,
            startup.configuration_fingerprint,
            startup.metadata_fingerprint,
            startup.capabilities,
            startup.operation_count,
            startup.protocol_version,
            startup.configuration_schema_version,
            environment,
            (),
            startup.log_mode,
            Status.OK,
        )
        sendmsg_strict(
            bootstrap, encode_startup(response, response=True, request_id=request_id)
        )
    except Exception as error:
        close_fds(descriptors)
        try:
            sendmsg_strict(
                bootstrap,
                encode_error(
                    ErrorResponse(Status.IDENTITY_MISMATCH, str(error)[:1024]),
                    request_id=request_id,
                ),
            )
        finally:
            bootstrap.close()
        raise

    mapped_buffers = MappedBufferCache()
    fd_receiver = FdReceiver(
        socket.socket(fileno=fd_socket_fd), mapped_buffers, startup.session_generation
    )
    fd_receiver.start()
    command_ring = RingEndpoint(*command_fds, startup.session_generation, False)
    completion_ring = RingEndpoint(*completion_fds, startup.session_generation, True)
    metadata = _metadata_from_declarations(operations)
    compiled = _compile_operations(metadata, operations)
    planners = _compile_planners(metadata, output_planners)
    ring_thread = threading.Thread(
        target=_ring_worker_loop,
        args=(
            command_ring,
            completion_ring,
            mapped_buffers,
            compiled,
            planners,
        ),
        daemon=True,
    )
    ring_thread.start()
    try:
        while True:
            packet, fds = recvmsg_strict(bootstrap)
            if fds:
                raise ValueError("lifecycle frame carried file descriptors")
            frame = decode_frame(packet)
            if frame.kind == Kind.PING:
                request = _empty_request(packet, Kind.PING)
                sendmsg_strict(bootstrap, encode_empty(Kind.PONG, request_id=request))
            elif frame.kind == Kind.SHUTDOWN:
                request = _empty_request(packet, Kind.SHUTDOWN)
                sendmsg_strict(
                    bootstrap, encode_empty(Kind.SHUTDOWN_ACK, request_id=request)
                )
                break
            else:
                raise ValueError("unexpected lifecycle frame")
    finally:
        bootstrap.close()
        command_ring.interrupt()
        completion_ring.interrupt()
        ring_thread.join(timeout=5)
        command_ring.close()
        completion_ring.close()
        fd_receiver.close()
        mapped_buffers.close()


def _empty_request(packet: bytes, kind: Kind) -> int:
    frame = decode_frame(packet)
    if frame.kind != kind or frame.payload:
        raise ValueError("invalid empty lifecycle frame")
    return frame.request_id


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _validate_startup_identity(
    startup: Startup, operations: Mapping[str, OperationHandler]
) -> None:
    required = {
        "configuration_fingerprint",
        "configuration_schema_version",
        "interface_fingerprint",
        "metadata_fingerprint",
        "operation_count",
        "protocol_version",
        "startup_capabilities",
    }
    if any(not hasattr(operations, name) for name in required):
        raise ValueError("operations were not bound from generated worker declarations")
    expected = (
        operations.interface_fingerprint,
        operations.configuration_fingerprint,
        operations.metadata_fingerprint,
        operations.startup_capabilities,
        operations.operation_count,
        operations.protocol_version,
        operations.configuration_schema_version,
    )
    actual = (
        startup.interface_fingerprint,
        startup.configuration_fingerprint,
        startup.metadata_fingerprint,
        startup.capabilities,
        startup.operation_count,
        startup.protocol_version,
        startup.configuration_schema_version,
    )
    if actual != expected:
        raise ValueError("startup manifest and generated worker identity differ")


def _metadata_from_declarations(
    handlers: Mapping[str, OperationHandler],
) -> tuple[OperationMetadata, ...]:
    declarations = getattr(handlers, "declarations", None)
    if not isinstance(declarations, (list, tuple)):
        raise ValueError("generated worker declarations are missing")
    return tuple(_operation_declaration(item) for item in declarations)


def _operation_declaration(item: dict[str, object]) -> OperationMetadata:
    outputs = item["outputs"]
    assert isinstance(outputs, (list, tuple))
    vjp_value = item["vjp"]
    vjp = (
        VjpMetadata(
            int(vjp_value["operation_id"]),
            tuple(vjp_value["saved_inputs"]),
            tuple(vjp_value["saved_outputs"]),
            tuple(vjp_value["output_cotangents"]),
            tuple(vjp_value["input_gradients"]),
            tuple(vjp_value["scalar_parameters"]),
        )
        if isinstance(vjp_value, dict)
        else None
    )
    return OperationMetadata(
        name=str(item["name"]),
        tensor_inputs=tuple(
            TensorParameter(
                str(value["name"]),
                "readWrite" if value["access"] == "read_write" else "readOnly",
                value["dtype_variable"],
                tuple(value["dtypes"]),
            )
            for value in item["inputs"]
        ),
        tensor_outputs=tuple(
            TensorParameter(str(value["name"]), "readOnly") for value in outputs
        ),
        scalar_parameters=tuple(
            ScalarParameter(
                str(value["name"]),
                str(value["kind"]),
                bool(value["required"]),
                value["default"],
                value["enum"],
                tuple(value["enum_values"]),
            )
            for value in item["scalars"]
        ),
        operation_id=int(item["operation_id"]),
        output_plans=tuple(_output_declaration(value) for value in outputs),
        vjp=vjp,
        internal=bool(item["internal"]),
        dtype_variables=tuple(
            DTypeVariable(str(value["name"]), tuple(value["dtypes"]))
            for value in item["dtype_variables"]
        ),
    )


def _output_declaration(value: dict[str, object]) -> OutputPlan:
    if value["allocation"] == "dynamic":
        return OutputPlan(str(value["name"]), None)
    dtype = value["dtype"]
    assert isinstance(dtype, dict)
    kind = str(dtype["kind"])
    if kind == "fixed":
        dtype_value: object = dtype["value"]
    elif kind == "input":
        dtype_value = int(dtype["input"])
    elif kind == "variable":
        dtype_value = dtype["variable"]
    else:
        dtype_value = PromoteTensorScalar(int(dtype["input"]), int(dtype["scalar"]))
    same = value["same_shape_as_input"]
    dimensions = value["dimensions"]
    known = KnownOutput(
        "sameShapeAsInput" if same is not None else "dimensions",
        int(same)
        if same is not None
        else tuple(_dimension_declaration(item) for item in dimensions),
        DTypeExpression(kind, dtype_value),
    )
    return OutputPlan(str(value["name"]), known)


def _dimension_declaration(value: dict[str, object]) -> DimensionExpression:
    kind = str(value["kind"])
    if kind == "constant":
        result: object = int(value["axis"])
    elif kind == "input_axis":
        result = InputAxis(int(value["input"]), int(value["axis"]))
    elif kind == "minimum":
        result = tuple(_dimension_declaration(item) for item in value["operands"])
    else:
        result = SelectDimension(
            int(value["scalar"]),
            _dimension_declaration(value["when_true"]),
            _dimension_declaration(value["when_false"]),
        )
    return DimensionExpression("inputAxis" if kind == "input_axis" else kind, result)


def _ring_worker_loop(
    commands: RingEndpoint,
    completions: RingEndpoint,
    mapped_buffers: MappedBufferCache,
    operations: Mapping[int, _Operation],
    planners: Mapping[int, OutputPlanner],
) -> None:
    while True:
        try:
            command = commands.pop()
        except RingError:
            return
        worker_dequeued_ns = perf_counter_ns() if command.flags & FLAG_PROFILE else 0
        completion_kind = {
            COMMAND_INVOKE: COMPLETION_INVOKE,
            COMMAND_PLAN_OUTPUTS: COMPLETION_PLAN_OUTPUTS,
            COMMAND_PING: COMPLETION_PONG,
        }.get(command.kind, COMPLETION_INVOKE)
        completion = Record(
            completion_kind,
            command.generation,
            command.submission_id,
            command.invocation_id,
            command.operation_id,
            STATUS_OK,
            flags=command.flags,
        )
        profile = list(completion.profile)
        profile[0] = command.profile[0]
        profile[1] = worker_dequeued_ns
        profile[2] = perf_counter_ns() if command.flags & FLAG_PROFILE else 0
        completion.profile = tuple(profile)
        try:
            if command.kind == COMMAND_INVOKE:
                measured = _invoke_known(
                    command.invocation(),
                    mapped_buffers,
                    operations,
                    profiled=bool(command.flags & FLAG_PROFILE),
                )
                if measured:
                    completion.profile = (
                        *completion.profile[:3],
                        measured["inputViewsNs"],
                        measured["outputViewsNs"],
                        measured["dispatchNs"],
                        measured["kernelNs"],
                        completion.profile[7],
                    )
            elif command.kind == COMMAND_PLAN_OUTPUTS:
                planned = _plan_outputs(
                    command.invocation(include_outputs=False),
                    mapped_buffers,
                    operations,
                    planners,
                )
                completion.outputs = tuple(
                    PlannedOutput(
                        int(item["output"]), tuple(item["shape"]), str(item["dtype"])
                    )
                    for item in planned
                )
            elif command.kind == COMMAND_PING:
                kernel_started = perf_counter_ns()
                if command.operation_id:
                    sleep(command.operation_id / 1_000_000_000)
                if command.flags & FLAG_PROFILE:
                    profile = list(completion.profile)
                    profile[6] = perf_counter_ns() - kernel_started
                    completion.profile = tuple(profile)
            else:
                raise ValueError("Unsupported ring command")
        except _OperationFailure as error:
            completion.status = STATUS_OPERATION_ERROR
            completion.error_type = error.error_type
            completion.error_message = error.message
        except Exception as error:
            completion.status = STATUS_INTERNAL_ERROR
            completion.error_type = type(error).__name__
            completion.error_message = str(error)
        try:
            if completion.flags & FLAG_PROFILE:
                profile = list(completion.profile)
                profile[7] = perf_counter_ns()
                completion.profile = tuple(profile)
            completions.push(completion)
        except RingError:
            return


class _OperationFailure(Exception):
    def __init__(self, error: Exception) -> None:
        self.error_type = type(error).__name__
        self.message = str(error)
        super().__init__(self.message)


def _compile_operations(
    metadata_operations: tuple[OperationMetadata, ...],
    handlers: Mapping[str, OperationHandler],
) -> dict[int, _Operation]:
    compiled = {}
    metadata_names = set()
    for metadata in metadata_operations:
        name = metadata.name
        metadata_names.add(name)
        try:
            handler = handlers[name]
        except KeyError:
            raise ValueError(f"Plugin has no handler for operation {name!r}") from None
        operation_id = metadata.operation_id
        compiled[operation_id] = _Operation(
            handler=handler,
            metadata=metadata,
            input_accesses=tuple(item.access for item in metadata.tensor_inputs),
            scalar_kinds=tuple(item.kind for item in metadata.scalar_parameters),
        )
    unknown = set(handlers) - metadata_names
    if unknown:
        raise ValueError(
            f"Handlers are not declared in plugin metadata: {sorted(unknown)}"
        )
    return compiled


def _compile_planners(
    metadata_operations: tuple[OperationMetadata, ...],
    planners: Mapping[str, OutputPlanner],
) -> dict[int, OutputPlanner]:
    dynamic = {
        operation.name: operation
        for operation in metadata_operations
        if any(plan.known is None for plan in operation.output_plans)
    }
    if set(planners) != set(dynamic):
        raise ValueError(
            "Dynamic output planners do not match metadata: "
            f"expected {sorted(dynamic)}, received {sorted(planners)}"
        )
    return {
        operation.operation_id: planners[name] for name, operation in dynamic.items()
    }


def _plan_outputs(
    invocation: object,
    mapped_buffers: MappedBufferCache,
    operations: Mapping[int, _Operation],
    planners: Mapping[int, OutputPlanner],
) -> list[dict[str, object]]:
    invocation_id = int(invocation.invocationId)
    operation_id = int(invocation.operationId)
    try:
        operation = operations[operation_id]
        planner = planners[operation_id]
    except KeyError:
        raise ValueError(
            f"Operation ID {operation_id} has no dynamic planner"
        ) from None
    if len(invocation.inputs) != len(operation.input_accesses):
        raise ValueError("Output planning has an invalid input count")
    inputs = tuple(
        mapped_buffers.tensor(descriptor, invocation_id=invocation_id)
        for descriptor in invocation.inputs
    )
    scalars = _decode_scalars(invocation.scalars, operation.scalar_kinds)
    context = InvocationContext(operation.metadata, invocation_id, inputs, (), scalars)
    try:
        planned = planner(context)
    except Exception as error:
        raise _OperationFailure(error) from error
    indices = {
        parameter.name: index
        for index, parameter in enumerate(operation.metadata.tensor_outputs)
        if operation.metadata.output_plans[index].known is None
    }
    if set(planned) != set(indices):
        raise _OperationFailure(
            ValueError("Planner returned the wrong dynamic outputs")
        )
    return [
        {
            "output": indices[name],
            "shape": list(spec.shape),
            "dtype": str(spec.dtype).removeprefix("torch."),
        }
        for name, spec in planned.items()
    ]


def _invoke_known(
    invocation: object,
    mapped_buffers: MappedBufferCache,
    operations: Mapping[int, _Operation],
    *,
    profiled: bool,
) -> dict[str, int] | None:
    invocation_id = int(invocation.invocationId)
    started = perf_counter_ns() if profiled else 0
    input_views_ns = 0
    output_views_ns = 0
    kernel_ns = 0
    try:
        operation_id = int(invocation.operationId)
        try:
            operation = operations[operation_id]
        except KeyError:
            raise ValueError(f"Unknown operation ID {operation_id}") from None
        if len(invocation.inputs) != len(operation.input_accesses):
            raise ValueError("Invocation has an invalid input count")
        if len(invocation.outputs) != len(operation.metadata.tensor_outputs):
            raise ValueError("Invocation has an invalid output count")

        view_started = perf_counter_ns() if profiled else 0
        inputs = tuple(
            mapped_buffers.tensor(
                descriptor,
                invocation_id=invocation_id,
                require_writable=access == "readWrite",
            )
            for descriptor, access in zip(
                invocation.inputs, operation.input_accesses, strict=True
            )
        )
        if profiled:
            input_views_ns = perf_counter_ns() - view_started
            view_started = perf_counter_ns()
        outputs = tuple(
            mapped_buffers.tensor(
                descriptor,
                invocation_id=invocation_id,
                require_writable=True,
            )
            for descriptor in invocation.outputs
        )
        if profiled:
            output_views_ns = perf_counter_ns() - view_started
        scalars = _decode_scalars(invocation.scalars, operation.scalar_kinds)
        context = InvocationContext(
            operation.metadata,
            invocation_id,
            inputs,
            outputs,
            scalars,
        )
        kernel_started = perf_counter_ns() if profiled else 0
        try:
            operation.handler(context)
        except Exception as error:
            raise _OperationFailure(error) from error
        kernel_ns = perf_counter_ns() - kernel_started if profiled else 0
        elapsed_ns = perf_counter_ns() - started if profiled else 0
    finally:
        mapped_buffers.finish_invocation(invocation_id)
    if not profiled:
        return None
    return {
        "inputViewsNs": input_views_ns,
        "outputViewsNs": output_views_ns,
        "dispatchNs": max(
            0,
            elapsed_ns - input_views_ns - output_views_ns - kernel_ns,
        ),
        "kernelNs": kernel_ns,
    }


def _decode_scalars(arguments: object, kinds: tuple[str, ...]) -> tuple[object, ...]:
    missing = object()
    values = [missing] * len(kinds)
    for argument in arguments:
        parameter = int(argument.parameter)
        if parameter >= len(kinds):
            raise TypeError("Scalar argument does not match operation metadata")
        if values[parameter] is not missing:
            raise ValueError("Scalar parameter was supplied more than once")
        kind = argument.which()
        if kinds[parameter] != kind:
            raise TypeError("Scalar argument does not match operation metadata")
        values[parameter] = getattr(argument, kind)
    if any(value is missing for value in values):
        raise ValueError("Invocation is missing a scalar argument")
    return tuple(values)


def _glibc_version() -> str:
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    version = libc.gnu_get_libc_version()
    if version is None:
        raise RuntimeError("glibc did not report a version")
    return version.decode()
