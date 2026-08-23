import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wmfs_tool.model import Operation, Plugin, ScalarParameter
from wmfs_tool.parser import load_interface

_GENERATOR = "wmfs-tool/2"


def generate(interface_path: Path, output: Path, *, check: bool = False) -> str:
    plugin = load_interface(interface_path)
    document = _interface_document(plugin)
    canonical = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    namespace = plugin.namespace.lower()
    python_path = f"python/{plugin.python_module}/interface.py"
    outputs = {
        output / "manifest.json": _manifest(plugin, document, fingerprint),
        output / "include/wmfs/plugin_abi.h": _abi_header(),
        output / f"include/wmfs/{namespace}_plugin.hpp": _cpp_wrapper(
            plugin, fingerprint
        ),
        output / f"src/{namespace}_plugin_stub.cpp": _cpp_stub(plugin),
        output / python_path: _python_metadata(plugin, fingerprint),
        output / f"python/{plugin.python_module}/interface.pyi": _python_stub(plugin),
        output / "reference_dispatch.inc": _worker_cpp_dispatch(plugin),
        output.parent / plugin.python_module / "_generated.py": _worker_python_adapter(
            plugin
        ),
    }
    stale = [path for path, content in outputs.items() if _read(path) != content]
    if check:
        if stale:
            raise RuntimeError(
                "generated artifacts are stale or missing: "
                + ", ".join(str(path) for path in stale)
            )
        return fingerprint
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    return fingerprint


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _interface_document(plugin: Plugin) -> dict[str, Any]:
    operations = []
    for operation in plugin.operations:
        item = asdict(operation)
        item["id"] = item.pop("operation_id")
        operations.append(item)
    return {
        "abiVersion": plugin.abi_version,
        "formatVersion": plugin.format_version,
        "operations": operations,
        "enums": [asdict(item) for item in plugin.enums],
        "plugin": {
            "name": plugin.name,
            "namespace": plugin.namespace,
            "pythonModule": plugin.python_module,
            "version": plugin.version,
        },
        "protocolVersion": plugin.protocol_version,
    }


def _manifest(plugin: Plugin, document: dict[str, Any], fingerprint: str) -> str:
    result = dict(document)
    result["deployment"] = {
        "interface": plugin.interface,
        "root": plugin.deployment_root,
        "schema": plugin.schema,
        "worker": plugin.worker,
    }
    result["generator"] = _GENERATOR
    result["interfaceFingerprint"] = f"sha256:{fingerprint}"
    result["metadataFingerprint"] = f"0x{_metadata_fingerprint(plugin):016x}"
    result["operationCount"] = len(plugin.operations)
    return (
        json.dumps(result, allow_nan=False, ensure_ascii=True, indent=4, sort_keys=True)
        + "\n"
    )


def _metadata_fingerprint(plugin: Plugin) -> int:
    metadata = {
        "name": plugin.name,
        "operations": [_metadata_operation(item) for item in plugin.operations],
        "protocol_version": plugin.protocol_version,
        "version": plugin.version,
        "metadata_version": 2,
    }
    envelope = {"encoding": "wmfs-plugin-metadata-v2", "metadata": metadata}
    canonical = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big")


def _metadata_operation(operation: Operation) -> dict[str, Any]:
    return {
        "internal": operation.internal,
        "name": operation.name,
        "operation_id": operation.operation_id,
        "output_plans": [_metadata_output(item) for item in operation.outputs],
        "scalar_parameters": [
            {
                "default": item.default,
                "enum_name": item.enum,
                "enum_values": item.enum_values,
                "kind": item.kind,
                "name": item.name,
                "required": item.required,
            }
            for item in operation.scalars
        ],
        "tensor_inputs": [
            {
                "access": _metadata_access(item.access),
                "dtype_variable": item.dtype_variable,
                "dtypes": item.dtypes,
                "name": item.name,
            }
            for item in operation.inputs
        ],
        "tensor_outputs": [
            {
                "access": "readOnly",
                "dtype_variable": None,
                "dtypes": (),
                "name": item.name,
            }
            for item in operation.outputs
        ],
        "vjp": asdict(operation.vjp) if operation.vjp is not None else None,
        "dtype_variables": [asdict(item) for item in operation.dtype_variables],
    }


def _metadata_access(access: str) -> str:
    return {"read_only": "readOnly", "read_write": "readWrite"}[access]


def _metadata_output(output: Any) -> dict[str, Any]:
    known = None
    if output.allocation == "known":
        if output.same_shape_as_input is not None:
            shape_kind = "sameShapeAsInput"
            shape: Any = output.same_shape_as_input
        else:
            shape_kind = "dimensions"
            shape = [_metadata_dimension(item) for item in output.dimensions]
        known = {
            "dtype": _metadata_dtype(output.dtype),
            "shape": shape,
            "shape_kind": shape_kind,
        }
    return {"known": known, "name": output.name}


def _metadata_dimension(dimension: Any) -> dict[str, Any]:
    if dimension.kind == "constant":
        value: Any = dimension.axis
    elif dimension.kind == "input_axis":
        value = {"axis": dimension.axis, "input": dimension.input}
    elif dimension.kind == "minimum":
        value = [_metadata_dimension(item) for item in dimension.operands]
    else:
        value = {
            "scalar_parameter": dimension.scalar,
            "when_false": _metadata_dimension(dimension.when_false),
            "when_true": _metadata_dimension(dimension.when_true),
        }
    kind = "inputAxis" if dimension.kind == "input_axis" else dimension.kind
    return {"kind": kind, "value": value}


def _metadata_dtype(dtype: Any) -> dict[str, Any]:
    if dtype.kind == "input":
        value: Any = dtype.input
    elif dtype.kind == "fixed":
        value = dtype.value
    elif dtype.kind == "variable":
        value = dtype.variable
    else:
        value = {"scalar_parameter": dtype.scalar, "tensor_input": dtype.input}
    kind = (
        "promoteTensorScalar" if dtype.kind == "promote_tensor_scalar" else dtype.kind
    )
    return {"kind": kind, "value": value}


def _abi_header() -> str:
    return """/* Generated by wmfs-tool. Do not edit. */
#ifndef WMFS_PLUGIN_ABI_H
#define WMFS_PLUGIN_ABI_H

#include <stdint.h>

#define WMFS_PLUGIN_ABI_VERSION UINT32_C(1)
#define WMFS_PLUGIN_MAX_RANK UINT32_C(16)
#define WMFS_PLUGIN_MAX_INPUTS UINT32_C(16)
#define WMFS_PLUGIN_MAX_OUTPUTS UINT32_C(8)

#ifdef __cplusplus
extern "C" {
#endif

typedef enum wmfs_status_v1 {
    WMFS_STATUS_OK = 0,
    WMFS_STATUS_INVALID_ARGUMENT = 1,
    WMFS_STATUS_UNSUPPORTED = 2,
    WMFS_STATUS_INTERNAL_ERROR = 3
} wmfs_status_v1;

typedef enum wmfs_dtype_v1 {
    WMFS_DTYPE_INVALID = 0,
    WMFS_DTYPE_BOOL = 1,
    WMFS_DTYPE_INT8 = 2,
    WMFS_DTYPE_UINT8 = 3,
    WMFS_DTYPE_INT16 = 4,
    WMFS_DTYPE_INT32 = 5,
    WMFS_DTYPE_INT64 = 6,
    WMFS_DTYPE_FLOAT16 = 7,
    WMFS_DTYPE_FLOAT32 = 8,
    WMFS_DTYPE_FLOAT64 = 9,
    WMFS_DTYPE_BFLOAT16 = 10
} wmfs_dtype_v1;

typedef enum wmfs_scalar_kind_v1 {
    WMFS_SCALAR_BOOLEAN = 1,
    WMFS_SCALAR_FLOAT64 = 2,
    WMFS_SCALAR_INT64 = 3,
    WMFS_SCALAR_TEXT = 4
} wmfs_scalar_kind_v1;

typedef struct wmfs_tensor_v1 {
    uint32_t struct_size;
    uint32_t flags;
    uint64_t buffer_id;
    uint64_t buffer_generation;
    uint64_t byte_offset;
    uint64_t byte_length;
    uint32_t dtype;
    uint32_t rank;
    int64_t shape[16];
    int64_t strides[16];
    void *data;
} wmfs_tensor_v1;

typedef struct wmfs_scalar_v1 {
    uint32_t struct_size;
    uint32_t parameter_index;
    uint32_t kind;
    uint32_t reserved;
    uint64_t bits;
    const char *text;
    uint64_t text_length;
} wmfs_scalar_v1;

typedef struct wmfs_invocation_v1 {
    uint32_t struct_size;
    uint32_t operation_id;
    uint64_t submission_id;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t scalar_count;
    uint32_t reserved;
    const wmfs_tensor_v1 *inputs;
    wmfs_tensor_v1 *outputs;
    const wmfs_scalar_v1 *scalars;
} wmfs_invocation_v1;

typedef int32_t (*wmfs_dispatch_v1)(const wmfs_invocation_v1 *invocation);

typedef struct wmfs_plugin_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t protocol_version;
    uint32_t reserved;
    const char *plugin_name;
    const char *plugin_version;
    const char *interface_fingerprint;
    wmfs_dispatch_v1 dispatch;
} wmfs_plugin_api_v1;

typedef const wmfs_plugin_api_v1 *(*wmfs_get_plugin_api_v1)(
    uint32_t abi_version);

#ifdef __cplusplus
}
#endif
#endif
"""


def _cpp_wrapper(plugin: Plugin, fingerprint: str) -> str:
    guard = f"WMFS_{plugin.namespace.upper()}_PLUGIN_HPP"
    fingerprint_macro = f"#define WMFS_{plugin.namespace.upper()}_INTERFACE_FINGERPRINT"
    fingerprint_padding = " " * max(1, 79 - len(fingerprint_macro))
    operation_lines = "\n".join(
        f"    {item.name} = UINT32_C({item.operation_id}),"
        for item in plugin.operations
    )
    enums = "\n\n".join(_cpp_enum(item.name, item.values) for item in plugin.enums)
    typed_declarations = "\n".join(
        f"template <typename T>\nstd::int32_t {item.name}_typed(dtype_tag<T>, const wmfs_invocation_v1 *);"
        for item in plugin.operations
    )
    return f"""// Generated by wmfs-tool. Do not edit.
#ifndef {guard}
#define {guard}

#include <cstdint>
#include <wmfs/plugin_abi.h>

#define WMFS_{plugin.namespace.upper()}_ABI_VERSION UINT32_C({plugin.abi_version})
#define WMFS_{plugin.namespace.upper()}_PROTOCOL_VERSION UINT32_C({plugin.protocol_version})
{fingerprint_macro}{fingerprint_padding}\\
    "sha256:{fingerprint}"

namespace wmfs {{
namespace {plugin.namespace} {{

enum operation_id : std::uint32_t {{
{operation_lines}
}};

{enums}

template <typename T> struct dtype_tag {{}};

{typed_declarations}

}} // namespace {plugin.namespace}
}} // namespace wmfs

extern "C" const wmfs_plugin_api_v1 *
wmfs_plugin_get_api(std::uint32_t abi_version);

#endif
"""


def _cpp_stub(plugin: Plugin) -> str:
    namespace = plugin.namespace.lower()
    cases = "\n".join(_dispatch_case(namespace, item) for item in plugin.operations)
    return f"""// Generated by wmfs-tool. Implement the declared operation functions.
#include <wmfs/{namespace}_plugin.hpp>

namespace {{
int32_t dispatch(const wmfs_invocation_v1 *invocation) {{
    if (invocation == 0 ||
        invocation->struct_size < sizeof(wmfs_invocation_v1)) {{
        return WMFS_STATUS_INVALID_ARGUMENT;
    }}
    switch (invocation->operation_id) {{
{cases}
    default:
        return WMFS_STATUS_UNSUPPORTED;
    }}
}}

const wmfs_plugin_api_v1 API = {{sizeof(wmfs_plugin_api_v1),
                                WMFS_{plugin.namespace.upper()}_ABI_VERSION,
                                WMFS_{plugin.namespace.upper()}_PROTOCOL_VERSION,
                                UINT32_C(0),
                                "{plugin.name}",
                                "{plugin.version}",
                                WMFS_{plugin.namespace.upper()}_INTERFACE_FINGERPRINT,
                                &dispatch}};
}} // namespace

extern "C" const wmfs_plugin_api_v1 *wmfs_plugin_get_api(uint32_t abi_version) {{
    return abi_version == WMFS_{plugin.namespace.upper()}_ABI_VERSION ? &API : 0;
}}
"""


def _dispatch_case(namespace: str, operation: Operation) -> str:
    validations = _cpp_dtype_validations(operation)
    dispatch = _cpp_typed_dispatch(namespace, operation)
    return f"""    case UINT32_C({operation.operation_id}):
        if (invocation->input_count != UINT32_C({len(operation.inputs)}) ||
            invocation->output_count != UINT32_C({len(operation.outputs)}) ||
            invocation->scalar_count != UINT32_C({len(operation.scalars)})) {{
            return WMFS_STATUS_INVALID_ARGUMENT;
        }}
{validations}
{dispatch}"""


_CPP_DTYPES = {
    "float32": ("WMFS_DTYPE_FLOAT32", "float"),
    "float64": ("WMFS_DTYPE_FLOAT64", "double"),
    "int64": ("WMFS_DTYPE_INT64", "std::int64_t"),
    "uint8": ("WMFS_DTYPE_UINT8", "std::uint8_t"),
}


def _operation_dtypes(operation: Operation) -> tuple[str, ...]:
    if operation.dtype_variables:
        return operation.dtype_variables[0].dtypes
    if operation.inputs and operation.inputs[0].dtypes:
        return operation.inputs[0].dtypes
    return tuple(_CPP_DTYPES)


def _cpp_dtype_validations(operation: Operation) -> str:
    lines = [
        "        if ((invocation->input_count && invocation->inputs == 0) ||",
        "            (invocation->output_count && invocation->outputs == 0) ||",
        "            (invocation->scalar_count && invocation->scalars == 0))",
        "            return WMFS_STATUS_INVALID_ARGUMENT;",
    ]
    variables = {
        item.name: index for index, item in enumerate(operation.dtype_variables)
    }
    first: dict[str, int] = {}
    for index, tensor in enumerate(operation.inputs):
        if tensor.dtype_variable is not None:
            variable = tensor.dtype_variable
            allowed = operation.dtype_variables[variables[variable]].dtypes
            expression = " &&\n            ".join(
                f"invocation->inputs[{index}].dtype != {_CPP_DTYPES[item][0]}"
                for item in allowed
            )
            lines.extend(
                (
                    f"        if ({expression})",
                    "            return WMFS_STATUS_UNSUPPORTED;",
                )
            )
            if variable in first:
                lines.extend(
                    (
                        f"        if (invocation->inputs[{index}].dtype != invocation->inputs[{first[variable]}].dtype)",
                        "            return WMFS_STATUS_INVALID_ARGUMENT;",
                    )
                )
            else:
                first[variable] = index
        elif tensor.dtypes:
            expression = " &&\n            ".join(
                f"invocation->inputs[{index}].dtype != {_CPP_DTYPES[item][0]}"
                for item in tensor.dtypes
            )
            lines.extend(
                (
                    f"        if ({expression})",
                    "            return WMFS_STATUS_UNSUPPORTED;",
                )
            )
    for index, scalar in enumerate(operation.scalars):
        if scalar.enum is not None:
            lines.extend(
                (
                    f"        if (invocation->scalars[{index}].kind != WMFS_SCALAR_INT64 ||",
                    f"            invocation->scalars[{index}].bits >= UINT64_C({len(scalar.enum_values)}))",
                    "            return WMFS_STATUS_INVALID_ARGUMENT;",
                )
            )
    variable_inputs = {
        item.dtype_variable: index
        for index, item in enumerate(operation.inputs)
        if item.dtype_variable is not None
    }
    for index, output in enumerate(operation.outputs):
        if output.dtype is None:
            continue
        if output.dtype.kind == "variable":
            expected = (
                f"invocation->inputs[{variable_inputs[output.dtype.variable]}].dtype"
            )
        elif output.dtype.kind == "input":
            expected = f"invocation->inputs[{output.dtype.input}].dtype"
        elif output.dtype.kind == "fixed":
            expected = _CPP_DTYPES[str(output.dtype.value)][0]
        else:
            continue
        lines.extend(
            (
                f"        if (invocation->outputs[{index}].dtype != {expected})",
                "            return WMFS_STATUS_INVALID_ARGUMENT;",
            )
        )
    return "\n".join(lines)


def _cpp_typed_dispatch(namespace: str, operation: Operation) -> str:
    index = 0
    lines = [f"        switch (invocation->inputs[{index}].dtype) {{"]
    for dtype in _operation_dtypes(operation):
        constant, cpp_type = _CPP_DTYPES[dtype]
        lines.extend(
            (
                f"        case {constant}:",
                f"            return wmfs::{namespace}::{operation.name}_typed(",
                f"                wmfs::{namespace}::dtype_tag<{cpp_type}>(), invocation);",
            )
        )
    lines.extend(
        ("        default:", "            return WMFS_STATUS_UNSUPPORTED;", "        }")
    )
    return "\n".join(lines)


def _cpp_enum(name: str, values: tuple[str, ...]) -> str:
    members = "\n".join(f"    {value} = {index}," for index, value in enumerate(values))
    return f"enum class {name} : std::int64_t {{\n{members}\n}};"


def _python_metadata(plugin: Plugin, fingerprint: str) -> str:
    operations = "\n".join(
        "\n".join(
            (
                "    Operation(",
                f"        {item.operation_id},",
                f"        {json.dumps(item.name)},",
                f"        {_python_tuple(value.name for value in item.inputs)},",
                f"        {_python_tuple(value.name for value in item.outputs)},",
                f"        {_python_tuple(value.name for value in item.scalars)},",
                "        "
                + _python_tuple(
                    value.name
                    for value in item.outputs
                    if value.allocation == "dynamic"
                )
                + ",",
                f"        {item.internal!r},",
                "    ),",
            )
        )
        for item in plugin.operations
    )
    enum_import = "from enum import IntEnum\n" if plugin.enums else ""
    enums = "\n\n\n".join(_python_enum(item.name, item.values) for item in plugin.enums)
    return f"""# Generated by wmfs-tool. Do not edit.
{enum_import}from types import MappingProxyType
from typing import NamedTuple

PLUGIN_NAME = {json.dumps(plugin.name)}
API_NAMESPACE = {json.dumps(plugin.namespace)}
PLUGIN_VERSION = {json.dumps(plugin.version)}
FORMAT_VERSION = {plugin.format_version}
ABI_VERSION = {plugin.abi_version}
PROTOCOL_VERSION = {plugin.protocol_version}
INTERFACE_FINGERPRINT = (
    "sha256:{fingerprint}"
)


{enums}


class Operation(NamedTuple):
    operation_id: int
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: tuple[str, ...]
    dynamic_outputs: tuple[str, ...]
    internal: bool


OPERATIONS = (
{operations}
)
OPERATIONS_BY_NAME = MappingProxyType({{item.name: item for item in OPERATIONS}})
"""


def _python_tuple(values: Iterable[str]) -> str:
    items = tuple(values)
    rendered = ", ".join(json.dumps(item) for item in items)
    if len(items) == 1:
        rendered += ","
    return f"({rendered})"


def _python_stub(plugin: Plugin) -> str:
    enums = "\n\n".join(
        _python_stub_enum(item.name, item.values) for item in plugin.enums
    )
    functions = "\n\n".join(
        _python_operation_stub(item) for item in plugin.operations if not item.internal
    )
    enum_import = "from enum import IntEnum\n" if plugin.enums else ""
    return f"""{enum_import}from typing import Final, Mapping, NamedTuple, overload

import torch

PLUGIN_NAME: Final[str]
API_NAMESPACE: Final[str]
PLUGIN_VERSION: Final[str]
FORMAT_VERSION: Final[int]
ABI_VERSION: Final[int]
PROTOCOL_VERSION: Final[int]
INTERFACE_FINGERPRINT: Final[str]

{enums}

class Operation(NamedTuple):
    operation_id: int
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: tuple[str, ...]
    dynamic_outputs: tuple[str, ...]
    internal: bool

OPERATIONS: tuple[Operation, ...]
OPERATIONS_BY_NAME: Mapping[str, Operation]

{functions}
"""


def _python_enum(name: str, values: tuple[str, ...]) -> str:
    members = "\n".join(f"    {value} = {index}" for index, value in enumerate(values))
    return f"class {name}(IntEnum):\n{members}"


def _python_stub_enum(name: str, values: tuple[str, ...]) -> str:
    members = "\n".join(f"    {value}: Final[{name}]" for value in values)
    return f"class {name}(IntEnum):\n{members}"


def _python_scalar_annotation(parameter: ScalarParameter) -> str:
    if parameter.enum is not None:
        return parameter.enum
    return {"boolean": "bool", "float64": "float", "int64": "int", "text": "str"}[
        parameter.kind
    ]


def _python_default(parameter: ScalarParameter) -> str:
    if parameter.required:
        return ""
    if parameter.enum is not None:
        return f" = {parameter.enum}.{parameter.default}"
    return f" = {parameter.default!r}"


def _python_operation_stub(operation: Operation) -> str:
    arguments = [
        f"{_python_name(item.name)}: torch.Tensor" for item in operation.inputs
    ]
    arguments.extend(
        f"{_python_name(item.name)}: {_python_scalar_annotation(item)}{_python_default(item)}"
        for item in operation.scalars
    )
    output_type = (
        "torch.Tensor"
        if len(operation.outputs) == 1
        else "tuple[" + ", ".join("torch.Tensor" for _ in operation.outputs) + "]"
    )
    out_type = output_type
    joined = ", ".join(arguments)
    separator = ", " if joined else ""
    return (
        "@overload\n"
        f"def {operation.name}({joined}{separator}*, out: None = None) -> {output_type}: ...\n"
        "@overload\n"
        f"def {operation.name}({joined}{separator}*, out: {out_type}) -> {out_type}: ..."
    )


def _python_name(name: str) -> str:
    return "".join(
        ("_" + character.lower()) if character.isupper() else character
        for character in name
    )


def _worker_python_adapter(plugin: Plugin) -> str:
    operations = "\n".join(
        "\n".join(
            (
                "    GeneratedOperation(",
                f"        {item.operation_id},",
                f"        {json.dumps(item.name)},",
                f"        {_python_tuple(value.name for value in item.inputs)},",
                f"        {_python_tuple(value.name for value in item.outputs)},",
                f"        {_python_tuple(value.name for value in item.scalars)},",
                "        "
                + _python_tuple(
                    value.name
                    for value in item.outputs
                    if value.allocation == "dynamic"
                )
                + ",",
                f"        {item.internal!r},",
                "    ),",
            )
        )
        for item in plugin.operations
    )
    adapters = "\n\n\n".join(
        _worker_python_operation(item) for item in plugin.operations
    )
    names = "\n".join(f"        {json.dumps(item.name)}," for item in plugin.operations)
    bindings = "\n".join(
        f"        {json.dumps(item.name)}: _adapt_{item.name}(implementations[{json.dumps(item.name)}]),"
        for item in plugin.operations
    )
    return f"""# Generated by wmfs-tool. Do not edit.
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from wmfs_plugin.invocation import InvocationContext

if TYPE_CHECKING:
    from wmfs_plugin.worker import OperationHandler
else:
    OperationHandler = Callable[[InvocationContext], None]

PLUGIN_NAME = {json.dumps(plugin.name)}
API_NAMESPACE = PLUGIN_NAME
PLUGIN_VERSION = {json.dumps(plugin.version)}
PROTOCOL_VERSION = {plugin.protocol_version}
METADATA_FINGERPRINT = 0x{_metadata_fingerprint(plugin):016X}


class GeneratedOperation(NamedTuple):
    operation_id: int
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: tuple[str, ...]
    dynamic_outputs: tuple[str, ...]
    internal: bool


OPERATIONS = (
{operations}
)
OPERATIONS_BY_NAME = MappingProxyType({{item.name: item for item in OPERATIONS}})
Implementation = Callable[..., None]


{adapters}


_OPERATION_NAMES = frozenset(
    {{
{names}
    }}
)


def bind_operations(
    implementations: Mapping[str, Implementation],
) -> dict[str, OperationHandler]:
    if set(implementations) != _OPERATION_NAMES:
        missing = sorted(_OPERATION_NAMES - set(implementations))
        unknown = sorted(set(implementations) - _OPERATION_NAMES)
        raise ValueError(
            f"Implementations do not match generated metadata: "
            f"missing={{missing}}, unknown={{unknown}}"
        )
    return {{
{bindings}
    }}
"""


def _worker_python_operation(operation: Operation) -> str:
    arguments = [
        *(f"context.inputs[{index}]" for index in range(len(operation.inputs))),
        *(f"context.scalars[{index}]" for index in range(len(operation.scalars))),
        *(f"context.outputs[{index}]" for index in range(len(operation.outputs))),
    ]
    joined = ", ".join(arguments)
    if len(joined) <= 70:
        call = f"        implementation({joined})"
    else:
        call = "\n".join(
            [
                "        implementation(",
                *(f"            {item}," for item in arguments),
                "        )",
            ]
        )
    return f"""def _adapt_{operation.name}(implementation: Implementation) -> OperationHandler:
    def handler(context: InvocationContext) -> None:
{call}

    return handler"""


_CPP_SCALARS = {
    "boolean": ("ScalarArgument::BOOLEAN", "getBoolean"),
    "float64": ("ScalarArgument::FLOAT64", "getFloat64"),
    "int64": ("ScalarArgument::INT64", "getInt64"),
    "text": ("ScalarArgument::TEXT", "getText"),
}


def _worker_cpp_dispatch(plugin: Plugin) -> str:
    cases = "\n".join(_worker_cpp_case(item) for item in plugin.operations)
    return f'''// Generated by wmfs-tool. Do not edit.
constexpr char WMFS_PLUGIN_VERSION[] = "{plugin.version}";
constexpr std::uint64_t WMFS_METADATA_FINGERPRINT = 0x{_metadata_fingerprint(plugin):016x}ULL;

void execute_known(std::uint32_t operation_id, std::vector<TensorLease> &inputs,
                   std::vector<TensorLease> &outputs,
                   capnp::List<ScalarArgument>::Reader scalars) {{
    switch (operation_id) {{
{cases}
    default:
        throw std::invalid_argument("Unknown operation ID " +
                                    std::to_string(operation_id));
    }}
}}
'''


def _worker_cpp_case(operation: Operation) -> str:
    lines = [
        f"    case {operation.operation_id}:",
        f"        require(inputs.size() == {len(operation.inputs)} && outputs.size() == {len(operation.outputs)} &&",
        f"                    scalars.size() == {len(operation.scalars)},",
        f'                "Invalid {operation.name} invocation");',
    ]
    cpp_dtypes = {
        "float32": "at::kFloat",
        "float64": "at::kDouble",
        "int64": "at::kLong",
        "uint8": "at::kByte",
    }
    variables = {item.name: item for item in operation.dtype_variables}
    bound_variables: dict[str, int] = {}
    for index, tensor in enumerate(operation.inputs):
        allowed = (
            variables[tensor.dtype_variable].dtypes
            if tensor.dtype_variable is not None
            else tensor.dtypes
        )
        if allowed:
            expression = " && ".join(
                f"inputs[{index}].tensor().scalar_type() != {cpp_dtypes[item]}"
                for item in allowed
            )
            lines.append(
                f'        require(!({expression}), "Tensor dtype is not supported by operation metadata");'
            )
        if tensor.dtype_variable is not None:
            if tensor.dtype_variable in bound_variables:
                previous = bound_variables[tensor.dtype_variable]
                lines.append(
                    f'        require(inputs[{index}].tensor().scalar_type() == inputs[{previous}].tensor().scalar_type(), "Tensor dtype variable does not match");'
                )
            else:
                bound_variables[tensor.dtype_variable] = index
    for index, scalar in enumerate(operation.scalars):
        enum_name, _getter = _CPP_SCALARS[scalar.kind]
        lines.extend(
            (
                f"        require(scalars[{index}].getParameter() == {index} &&",
                f"                    scalars[{index}].which() == {enum_name},",
                '                "Scalar argument does not match operation metadata");',
            )
        )
        if scalar.enum is not None:
            lines.extend(
                (
                    f"        require(scalars[{index}].getInt64() >= 0 &&",
                    f"                    scalars[{index}].getInt64() < {len(scalar.enum_values)},",
                    '                "Enum scalar is outside its declared range");',
                )
            )
    arguments = [
        *(f"inputs[{index}].tensor()" for index in range(len(operation.inputs))),
        *(
            f"scalars[{index}].{_CPP_SCALARS[scalar.kind][1]}()"
            for index, scalar in enumerate(operation.scalars)
        ),
        *(f"outputs[{index}].tensor()" for index in range(len(operation.outputs))),
    ]
    prefix = f"        {operation.name}_out("
    current = prefix
    continuation = " " * len(prefix)
    for index, argument in enumerate(arguments):
        token = argument + (");" if index == len(arguments) - 1 else ",")
        separator = "" if current == prefix else " "
        if len(current) + len(separator) + len(token) > 80:
            lines.append(current)
            current = continuation + token
        else:
            current += separator + token
    lines.extend((current, "        return;"))
    return "\n".join(lines)
