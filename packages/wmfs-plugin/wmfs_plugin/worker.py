import argparse
import asyncio
import ctypes
import socket
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType
from typing import TypeAlias

import capnp
import torch

from wmfs_plugin.fd_transport import FdReceiver, MappedBufferCache
from wmfs_plugin.invocation import InvocationContext, OutputSpec
from wmfs_plugin.metadata import OperationMetadata, metadata_from_reader
from wmfs_plugin.schema import PROTOCOL_VERSION, load_tensor_schema, schema_root

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
        The worker entry point is launched by WMFS and receives its RPC socket,
        FD-control socket, schema, and interface through command-line options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-fd", type=int, required=True)
    parser.add_argument("--fd-socket-fd", type=int, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument(
        "--schema-import", type=Path, action="append", default=[], required=True
    )
    arguments = parser.parse_args()
    asyncio.run(
        capnp.run(
            _serve(
                arguments.rpc_fd,
                arguments.fd_socket_fd,
                arguments.schema,
                arguments.schema_import,
                arguments.interface,
                operations,
                output_planners or {},
            )
        )
    )


async def _serve(
    rpc_fd: int,
    fd_socket_fd: int,
    plugin_schema_path: Path,
    schema_import_paths: list[Path],
    interface_name: str,
    operations: Mapping[str, OperationHandler],
    output_planners: Mapping[str, OutputPlanner],
) -> None:
    imports = [schema_root(), *schema_import_paths]
    plugin_schema = capnp.load(
        str(plugin_schema_path), imports=[str(path) for path in dict.fromkeys(imports)]
    )
    mapped_buffers = MappedBufferCache()
    fd_receiver = FdReceiver(
        socket.socket(fileno=fd_socket_fd), load_tensor_schema(), mapped_buffers
    )
    fd_receiver.start()
    rpc_socket = socket.socket(fileno=rpc_fd)
    stream = await capnp.AsyncIoStream.create_unix_connection(sock=rpc_socket)
    server = capnp.TwoPartyServer(
        stream,
        bootstrap=_make_server(
            plugin_schema,
            interface_name,
            mapped_buffers,
            operations,
            output_planners,
        ),
    )
    try:
        await server.on_disconnect()
    finally:
        server.close()
        stream.close()
        fd_receiver.close()
        mapped_buffers.close()


def _make_server(
    plugin_schema: ModuleType,
    interface_name: str,
    mapped_buffers: MappedBufferCache,
    handlers: Mapping[str, OperationHandler],
    output_planners: Mapping[str, OutputPlanner] | None = None,
) -> object:
    metadata = plugin_schema.pluginMetadata
    if int(metadata.protocolVersion) != PROTOCOL_VERSION:
        raise RuntimeError("Worker schema does not match its protocol version")
    parsed_metadata = metadata_from_reader(metadata)
    operations = _compile_operations(parsed_metadata.operations, handlers)
    planners = _compile_planners(parsed_metadata.operations, output_planners or {})
    interface = getattr(plugin_schema, interface_name)

    class PluginServer(interface.Server):
        async def getMetadata(self, _context: object, **_kwargs: object) -> object:
            return metadata

        async def getProtocolVersion(
            self, _context: object, **_kwargs: object
        ) -> tuple[int]:
            return (PROTOCOL_VERSION,)

        async def ping(
            self, nonce: int, _context: object, **_kwargs: object
        ) -> tuple[int]:
            return (nonce,)

        async def getEnvironment(
            self, _context: object, **_kwargs: object
        ) -> tuple[dict[str, str]]:
            return (
                {
                    "pythonVersion": sys.version.split()[0],
                    "torchVersion": torch.__version__,
                    "glibcVersion": _glibc_version(),
                    "executable": sys.executable,
                },
            )

        async def invokeKnown(
            self,
            invocation: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[dict[str, object]]:
            try:
                _invoke_known(invocation, mapped_buffers, operations, profiled=False)
            except _OperationFailure as error:
                return ({"operationError": error.as_capnp()},)
            return ({"success": None},)

        async def invokeKnownProfiled(
            self,
            invocation: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[dict[str, object], dict[str, int]]:
            try:
                metrics = _invoke_known(
                    invocation, mapped_buffers, operations, profiled=True
                )
            except _OperationFailure as error:
                return ({"operationError": error.as_capnp()}, {})
            assert metrics is not None
            return ({"success": None}, metrics)

        async def planOutputs(
            self,
            invocation: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[dict[str, object], list[dict[str, object]]]:
            try:
                outputs = _plan_outputs(
                    invocation, mapped_buffers, operations, planners
                )
            except _OperationFailure as error:
                return ({"operationError": error.as_capnp()}, [])
            return ({"success": None}, outputs)

    return PluginServer()


class _OperationFailure(Exception):
    def __init__(self, error: Exception) -> None:
        self.error_type = type(error).__name__
        self.message = str(error)
        super().__init__(self.message)

    def as_capnp(self) -> dict[str, str]:
        return {"type": self.error_type, "message": self.message}


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
