import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from wmfs_tool.model import ConfigurationProperty, Operation, Plugin, ScalarParameter
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
        "lifecycle": asdict(plugin.lifecycle),
    }


def _manifest(plugin: Plugin, document: dict[str, Any], fingerprint: str) -> str:
    result = dict(document)
    result["deployment"] = {
        "entrySymbol": f"wmfs_{plugin.namespace.lower()}_plugin_get_api",
        "root": plugin.deployment_root,
        "worker": plugin.worker,
    }
    if plugin.python_provider is not None:
        # Manifest v2 retains the legacy wire spelling for compatibility.
        result["deployment"]["localProvider"] = plugin.python_provider
    if plugin.bundled_namespace is not None:
        result["deployment"]["bundledNamespace"] = plugin.bundled_namespace
    result["generator"] = _GENERATOR
    result["controlAbiVersion"] = 1
    result["interfaceFingerprint"] = f"sha256:{fingerprint}"
    result["metadataFingerprint"] = f"0x{_metadata_fingerprint(plugin):016x}"
    result["operationCount"] = len(plugin.operations)
    result["startupCapabilities"] = _startup_capabilities(plugin)
    result["configuration"] = _configuration_manifest(plugin)
    return (
        json.dumps(result, allow_nan=False, ensure_ascii=True, indent=4, sort_keys=True)
        + "\n"
    )


def _startup_capabilities(plugin: Plugin) -> int:
    capabilities = (1 << 0) | (1 << 1) | (1 << 2)
    if plugin.configuration is not None:
        capabilities |= 1 << 3
    capabilities |= (1 << 4) | (1 << 5)
    if plugin.lifecycle.initialize:
        capabilities |= 1 << 6
    if plugin.lifecycle.shutdown:
        capabilities |= 1 << 7
    return capabilities


def _digest_initializer(value: str | None) -> str:
    digest = bytes.fromhex(value.removeprefix("sha256:")) if value else bytes(32)
    lines = [
        "    " + ", ".join(f"0x{item:02x}" for item in digest[index : index + 11])
        for index in range(0, len(digest), 11)
    ]
    return "{\n" + ",\n".join(lines) + "}"


def _configuration_manifest(plugin: Plugin) -> dict[str, Any] | None:
    if plugin.configuration is None:
        return None
    schema = _configuration_schema(plugin.configuration)
    return {
        "encoding": "wmfs-configuration-schema-v1",
        "examples": {name: value for name, value in plugin.configuration.examples},
        "fingerprint": _configuration_fingerprint(plugin),
        "schema": schema,
        "schemaVersion": plugin.configuration.schema_version,
    }


def _configuration_fingerprint(plugin: Plugin) -> str | None:
    if plugin.configuration is None:
        return None
    envelope = {
        "encoding": "wmfs-configuration-schema-v1",
        "schema": _configuration_schema(plugin.configuration),
    }
    canonical = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def _configuration_schema(configuration: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "additionalProperties": False,
        "properties": {
            item.name: _configuration_property_document(item)
            for item in configuration.properties
        },
        "required": [item.name for item in configuration.properties if item.required],
        "type": "object",
    }
    if configuration.description is not None:
        result["description"] = configuration.description
    return result


def _configuration_property_document(item: ConfigurationProperty) -> dict[str, Any]:
    result: dict[str, Any] = {"type": item.kind}
    if item.description is not None:
        result["description"] = item.description
    if item.has_default:
        result["default"] = item.default
    if item.enum:
        result["enum"] = list(item.enum)
    for model_name, document_name in (
        ("minimum", "minimum"),
        ("maximum", "maximum"),
        ("min_length", "minLength"),
        ("max_length", "maxLength"),
        ("min_items", "minItems"),
        ("max_items", "maxItems"),
    ):
        value = getattr(item, model_name)
        if value is not None:
            result[document_name] = value
    if item.kind == "object":
        result.update(
            {
                "additionalProperties": False,
                "properties": {
                    child.name: _configuration_property_document(child)
                    for child in item.properties
                },
                "required": [child.name for child in item.properties if child.required],
            }
        )
    elif item.kind == "array":
        assert item.items is not None
        result["items"] = _configuration_property_document(item.items)
    return result


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
#define WMFS_PLUGIN_MAX_LOG_FIELDS UINT32_C(32)
#define WMFS_PLUGIN_FEATURE_INITIALIZE (UINT64_C(1) << 0)
#define WMFS_PLUGIN_FEATURE_SHUTDOWN (UINT64_C(1) << 1)

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

typedef enum wmfs_log_level_v1 {
    WMFS_LOG_DEBUG = 10,
    WMFS_LOG_INFO = 20,
    WMFS_LOG_WARNING = 30,
    WMFS_LOG_ERROR = 40,
    WMFS_LOG_CRITICAL = 50
} wmfs_log_level_v1;

typedef enum wmfs_log_field_kind_v1 {
    WMFS_LOG_FIELD_BOOLEAN = 1,
    WMFS_LOG_FIELD_INT64 = 2,
    WMFS_LOG_FIELD_UINT64 = 3,
    WMFS_LOG_FIELD_FLOAT64 = 4,
    WMFS_LOG_FIELD_TEXT = 5
} wmfs_log_field_kind_v1;

typedef struct wmfs_text_view_v1 {
    const char *data;
    uint64_t size;
} wmfs_text_view_v1;

typedef wmfs_text_view_v1 wmfs_json_view_v1;

typedef struct wmfs_log_field_v1 {
    uint32_t struct_size;
    uint32_t kind;
    wmfs_text_view_v1 name;
    uint64_t bits;
    wmfs_text_view_v1 text;
} wmfs_log_field_v1;

typedef uint8_t (*wmfs_log_enabled_v1)(void *context, uint32_t level);
typedef void (*wmfs_log_write_v1)(void *context, uint32_t level,
                                  wmfs_text_view_v1 message,
                                  wmfs_text_view_v1 category,
                                  const wmfs_log_field_v1 *fields,
                                  uint32_t field_count);

typedef struct wmfs_logger_v1 {
    uint32_t struct_size;
    uint32_t reserved;
    void *context;
    wmfs_log_enabled_v1 enabled;
    wmfs_log_write_v1 log;
} wmfs_logger_v1;

typedef struct wmfs_error_buffer_v1 {
    uint32_t struct_size;
    uint32_t capacity;
    char *data;
    uint32_t size;
    uint32_t truncated;
} wmfs_error_buffer_v1;

typedef struct wmfs_initialize_args_v1 {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t features;
    wmfs_json_view_v1 configuration;
    wmfs_logger_v1 logger;
    wmfs_error_buffer_v1 *error;
} wmfs_initialize_args_v1;

typedef int32_t (*wmfs_initialize_v1)(const wmfs_initialize_args_v1 *args);
typedef void (*wmfs_shutdown_v1)(void);

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

typedef struct wmfs_output_plan_v1 {
    uint32_t struct_size;
    uint32_t output_index;
    uint32_t dtype;
    uint32_t rank;
    int64_t shape[16];
} wmfs_output_plan_v1;

typedef int32_t (*wmfs_plan_outputs_v1)(const wmfs_invocation_v1 *invocation,
                                        wmfs_output_plan_v1 *outputs,
                                        uint32_t output_capacity,
                                        uint32_t *output_count);

typedef struct wmfs_plugin_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t protocol_version;
    uint32_t reserved;
    const char *plugin_name;
    const char *plugin_version;
    const char *interface_fingerprint;
    wmfs_dispatch_v1 dispatch;
    wmfs_plan_outputs_v1 plan_outputs;
    uint64_t features;
    wmfs_initialize_v1 initialize;
    wmfs_shutdown_v1 shutdown;
} wmfs_plugin_api_v1;

typedef const wmfs_plugin_api_v1 *(*wmfs_get_plugin_api_v1)(
    uint32_t abi_version);

#ifdef __cplusplus
}
#endif
#endif
"""


def _cpp_wrapper(plugin: Plugin, fingerprint: str) -> str:
    namespace = plugin.namespace.lower()
    guard = f"WMFS_{plugin.namespace.upper()}_PLUGIN_HPP"
    fingerprint_macro = f"#define WMFS_{plugin.namespace.upper()}_INTERFACE_FINGERPRINT"
    fingerprint_padding = " " * max(1, 79 - len(fingerprint_macro))
    configuration_macro = (
        f"#define WMFS_{plugin.namespace.upper()}_CONFIGURATION_FINGERPRINT"
    )
    configuration_padding = " " * max(1, 79 - len(configuration_macro))
    operation_lines = "\n".join(
        f"    {item.name} = UINT32_C({item.operation_id}),"
        for item in plugin.operations
    )
    enums = "\n\n".join(_cpp_enum(item.name, item.values) for item in plugin.enums)
    typed_declarations = "\n".join(
        f"template <typename T>\nstd::int32_t {item.name}_typed(dtype_tag<T>, const wmfs_invocation_v1 *);"
        for item in plugin.operations
    )
    planner_declarations = "\n".join(
        f"template <typename T>\nstd::int32_t {item.name}_plan_typed(dtype_tag<T>, const wmfs_invocation_v1 *,\n"
        "                                wmfs_output_plan_v1 *, std::uint32_t,\n"
        "                                std::uint32_t *);"
        for item in plugin.operations
        if any(output.allocation == "dynamic" for output in item.outputs)
    )
    configuration_fingerprint = _configuration_fingerprint(plugin)
    configuration_version = (
        plugin.configuration.schema_version if plugin.configuration else 0
    )
    configuration_declarations = _cpp_configuration_declarations(plugin)
    metadata_declarations = "\n\n".join(
        item for item in (enums, configuration_declarations) if item
    )
    operation_declarations = "\n\n".join(
        item for item in (typed_declarations, planner_declarations) if item
    )
    metadata_block = f"\n\n{metadata_declarations}" if metadata_declarations else ""
    configuration_fingerprint_declaration = (
        f"{configuration_macro}{configuration_padding}\\\n"
        f"    {json.dumps(configuration_fingerprint)}"
        if configuration_fingerprint
        else f'{configuration_macro} ""'
    )
    lifecycle_declarations = ""
    if plugin.lifecycle.initialize:
        lifecycle_declarations += (
            "\nstd::int32_t initialize(wmfs_json_view_v1 configuration, "
            "logger log,\n                        wmfs_error_buffer_v1 *error);\n"
        )
    if plugin.lifecycle.shutdown:
        lifecycle_declarations += "\nvoid shutdown();\n"
    lifecycle_declarations = lifecycle_declarations.strip()
    lifecycle_block = "\n\n" + lifecycle_declarations if lifecycle_declarations else ""
    return f"""// Generated by wmfs-tool. Do not edit.
#ifndef {guard}
#define {guard}

#include <cstdint>
#include <wmfs/plugin_abi.h>

#define WMFS_{plugin.namespace.upper()}_ABI_VERSION UINT32_C({plugin.abi_version})
#define WMFS_{plugin.namespace.upper()}_PROTOCOL_VERSION UINT32_C({plugin.protocol_version})
#define WMFS_{plugin.namespace.upper()}_PLUGIN_VERSION {json.dumps(plugin.version)}
#define WMFS_{plugin.namespace.upper()}_CONFIGURATION_SCHEMA_VERSION UINT32_C({configuration_version})
#define WMFS_{plugin.namespace.upper()}_HAS_INITIALIZE {int(plugin.lifecycle.initialize)}
#define WMFS_{plugin.namespace.upper()}_HAS_SHUTDOWN {int(plugin.lifecycle.shutdown)}
#define WMFS_{plugin.namespace.upper()}_METADATA_FINGERPRINT UINT64_C(0x{_metadata_fingerprint(plugin):016x})
#define WMFS_{plugin.namespace.upper()}_OPERATION_COUNT UINT32_C({len(plugin.operations)})
#define WMFS_{plugin.namespace.upper()}_STARTUP_CAPABILITIES UINT64_C({_startup_capabilities(plugin)})
{fingerprint_macro}{fingerprint_padding}\\
    "sha256:{fingerprint}"
{configuration_fingerprint_declaration}

namespace wmfs {{
namespace {plugin.namespace} {{

static const std::uint8_t interface_fingerprint_sha256[32] = {_digest_initializer("sha256:" + fingerprint)};
static const std::uint8_t configuration_fingerprint_sha256[32] = {_digest_initializer(configuration_fingerprint)};

class logger {{
  public:
    logger() : value_(0) {{}}
    explicit logger(const wmfs_logger_v1 *value) : value_(value) {{}}

    bool enabled(std::uint32_t level) const {{
        return value_ && value_->enabled &&
               value_->enabled(value_->context, level);
    }}
    void log(std::uint32_t level, const char *message, std::uint64_t size,
             const char *category = 0, std::uint64_t category_size = 0,
             const wmfs_log_field_v1 *fields = 0,
             std::uint32_t field_count = 0) const {{
        if (!enabled(level) || !value_->log)
            return;
        const wmfs_text_view_v1 message_view = {{message, size}};
        const wmfs_text_view_v1 category_view = {{category, category_size}};
        value_->log(value_->context, level, message_view, category_view, fields,
                    field_count);
    }}
    void debug(const char *message, std::uint64_t size) const {{
        log(WMFS_LOG_DEBUG, message, size);
    }}
    void info(const char *message, std::uint64_t size) const {{
        log(WMFS_LOG_INFO, message, size);
    }}
    void warning(const char *message, std::uint64_t size) const {{
        log(WMFS_LOG_WARNING, message, size);
    }}
    void error(const char *message, std::uint64_t size) const {{
        log(WMFS_LOG_ERROR, message, size);
    }}
    void critical(const char *message, std::uint64_t size) const {{
        log(WMFS_LOG_CRITICAL, message, size);
    }}
    const wmfs_logger_v1 *native_handle() const {{ return value_; }}

  private:
    const wmfs_logger_v1 *value_;
}};

enum class operation_id : std::uint32_t {{
{operation_lines}
}};{metadata_block}

template <typename T> struct dtype_tag {{}};

{operation_declarations}{lifecycle_block}

}} // namespace {plugin.namespace}
}} // namespace wmfs

extern "C" const wmfs_plugin_api_v1 *
wmfs_{namespace}_plugin_get_api(std::uint32_t abi_version);

#endif
"""


def _cpp_configuration_declarations(plugin: Plugin) -> str:
    if plugin.configuration is None:
        return ""
    lines = ["namespace configuration {"]

    def visit(
        properties: tuple[ConfigurationProperty, ...], path: tuple[str, ...]
    ) -> None:
        for item in properties:
            constant = "_".join((*path, item.name))
            lines.append(f"constexpr char {constant}_key[] = {json.dumps(item.name)};")
            if (
                item.kind == "string"
                and item.enum
                and all(isinstance(value, str) for value in item.enum)
            ):
                enum_name = "".join(
                    part[:1].upper() + part[1:] for part in (*path, item.name)
                )
                members = "\n".join(
                    f"    {value} = {index}," for index, value in enumerate(item.enum)
                )
                lines.append(
                    f"enum class {enum_name} : std::uint32_t {{\n{members}\n}};"
                )
            visit(item.properties, (*path, item.name))

    visit(plugin.configuration.properties, ())
    lines.append("} // namespace configuration")
    return "\n\n".join(lines)


def _cpp_stub(plugin: Plugin) -> str:
    namespace = plugin.namespace.lower()
    cases = "\n".join(_dispatch_case(namespace, item) for item in plugin.operations)
    planner_cases = "\n".join(
        _planner_case(namespace, item)
        for item in plugin.operations
        if any(output.allocation == "dynamic" for output in item.outputs)
    )
    unused_output_capacity = "    (void)output_capacity;\n" if not planner_cases else ""
    features = (1 if plugin.lifecycle.initialize else 0) | (
        2 if plugin.lifecycle.shutdown else 0
    )
    initialize_adapter = (
        f"""int32_t initialize(const wmfs_initialize_args_v1 *args) {{
    if (args == 0 || args->struct_size < sizeof(wmfs_initialize_args_v1))
        return WMFS_STATUS_INVALID_ARGUMENT;
    try {{
        return wmfs::{namespace}::initialize(
            args->configuration, wmfs::{namespace}::logger(&args->logger),
            args->error);
    }} catch (...) {{
        return WMFS_STATUS_INTERNAL_ERROR;
    }}
}}
"""
        if plugin.lifecycle.initialize
        else ""
    )
    shutdown_adapter = (
        f"""void shutdown() {{
    try {{
        wmfs::{namespace}::shutdown();
    }} catch (...) {{
    }}
}}
"""
        if plugin.lifecycle.shutdown
        else ""
    )
    initialize_pointer = "&initialize" if plugin.lifecycle.initialize else "0"
    shutdown_pointer = "&shutdown" if plugin.lifecycle.shutdown else "0"
    return f"""// Generated by wmfs-tool. Implement the declared operation functions.
#include <wmfs/{namespace}_plugin.hpp>

namespace {{
{initialize_adapter}{shutdown_adapter}
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

int32_t plan_outputs(const wmfs_invocation_v1 *invocation,
                     wmfs_output_plan_v1 *outputs, uint32_t output_capacity,
                     uint32_t *output_count) {{
{unused_output_capacity}    if (invocation == 0 || outputs == 0 || output_count == 0 ||
        invocation->struct_size < sizeof(wmfs_invocation_v1)) {{
        return WMFS_STATUS_INVALID_ARGUMENT;
    }}
    switch (invocation->operation_id) {{
{planner_cases}
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
                                &dispatch,
                                &plan_outputs,
                                UINT64_C({features}),
                                {initialize_pointer},
                                {shutdown_pointer}}};
}} // namespace

extern "C" const wmfs_plugin_api_v1 *
wmfs_{namespace}_plugin_get_api(uint32_t abi_version) {{
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


def _planner_case(namespace: str, operation: Operation) -> str:
    dispatch = (
        _cpp_typed_dispatch(namespace, operation)
        .replace(f"::{operation.name}_typed(", f"::{operation.name}_plan_typed(")
        .replace("invocation);", "invocation, outputs, output_capacity, output_count);")
    )
    dispatch = dispatch.replace(
        "invocation, outputs, output_capacity, output_count);",
        "invocation, outputs,\n                output_capacity, output_count);",
    )
    return f"""    case UINT32_C({operation.operation_id}):
        if (invocation->input_count != UINT32_C({len(operation.inputs)}) ||
            invocation->output_count != UINT32_C(0) ||
            invocation->scalar_count != UINT32_C({len(operation.scalars)}) ||
            (invocation->input_count && invocation->inputs == 0) ||
            (invocation->scalar_count && invocation->scalars == 0)) {{
            return WMFS_STATUS_INVALID_ARGUMENT;
        }}
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
    configuration = _configuration_manifest(plugin)
    configuration_schema = configuration["schema"] if configuration else None
    configuration_examples = configuration["examples"] if configuration else {}
    configuration_import = "import json\n" if configuration else ""
    configuration_schema_version = (
        plugin.configuration.schema_version if plugin.configuration else None
    )
    configuration_fingerprint = _configuration_fingerprint(plugin)
    configuration_fingerprint_literal = (
        "(\n    " + json.dumps(configuration_fingerprint) + "\n)"
        if configuration_fingerprint
        else "None"
    )
    schema_literal = _python_json_literal(configuration_schema)
    examples_literal = (
        _python_json_literal(configuration_examples, "    ")
        if configuration
        else "    {}"
    )
    examples_declaration = (
        f"CONFIGURATION_EXAMPLES = MappingProxyType(\n{examples_literal}\n)"
        if configuration
        else "CONFIGURATION_EXAMPLES = MappingProxyType({})"
    )
    enums_block = f"\n\n\n{enums}" if enums else ""
    return f"""# Generated by wmfs-tool. Do not edit.
{configuration_import}{enum_import}from types import MappingProxyType
from typing import NamedTuple

PLUGIN_NAME = {json.dumps(plugin.name)}
API_NAMESPACE = {json.dumps(plugin.namespace)}
PLUGIN_VERSION = {json.dumps(plugin.version)}
FORMAT_VERSION = {plugin.format_version}
ABI_VERSION = {plugin.abi_version}
PROTOCOL_VERSION = {plugin.protocol_version}
OPERATION_COUNT = {len(plugin.operations)}
STARTUP_CAPABILITIES = {_startup_capabilities(plugin)}
HAS_INITIALIZE = {plugin.lifecycle.initialize!r}
HAS_SHUTDOWN = {plugin.lifecycle.shutdown!r}
INTERFACE_FINGERPRINT = (
    "sha256:{fingerprint}"
)
INTERFACE_FINGERPRINT_SHA256 = bytes.fromhex(
    {json.dumps(fingerprint)}
)
CONFIGURATION_SCHEMA_VERSION = {configuration_schema_version!r}
CONFIGURATION_FINGERPRINT = {configuration_fingerprint_literal}
CONFIGURATION_FINGERPRINT_SHA256 = bytes.fromhex(
    {json.dumps(configuration_fingerprint.removeprefix("sha256:") if configuration_fingerprint else "00" * 32)}
)
CONFIGURATION_SCHEMA = {schema_literal}
{examples_declaration}{enums_block}


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


def _python_json_literal(value: Any, indentation: str = "") -> str:
    if value is None:
        return indentation + "None"
    canonical = json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return "\n".join(
        (
            indentation + "json.loads(",
            indentation + "    " + repr(canonical),
            indentation + ")",
        )
    )


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
    functions = "\n".join(
        _python_operation_stub(item) for item in plugin.operations if not item.internal
    )
    enum_import = "from enum import IntEnum\n" if plugin.enums else ""
    configuration_types = _python_configuration_types(plugin)
    typing_names = ["Final", "Mapping", "NamedTuple", "overload"]
    if plugin.configuration is not None:
        typing_names.extend(("Literal", "TypedDict"))
    enums_block = f"{enums}\n\n" if enums else ""
    return f"""{enum_import}from typing import {", ".join(sorted(typing_names))}

import torch

PLUGIN_NAME: Final[str]
API_NAMESPACE: Final[str]
PLUGIN_VERSION: Final[str]
FORMAT_VERSION: Final[int]
ABI_VERSION: Final[int]
PROTOCOL_VERSION: Final[int]
OPERATION_COUNT: Final[int]
STARTUP_CAPABILITIES: Final[int]
HAS_INITIALIZE: Final[bool]
HAS_SHUTDOWN: Final[bool]
INTERFACE_FINGERPRINT: Final[str]
INTERFACE_FINGERPRINT_SHA256: Final[bytes]
CONFIGURATION_SCHEMA_VERSION: Final[int | None]
CONFIGURATION_FINGERPRINT: Final[str | None]
CONFIGURATION_FINGERPRINT_SHA256: Final[bytes]
CONFIGURATION_SCHEMA: Final[Mapping[str, object] | None]
CONFIGURATION_EXAMPLES: Final[Mapping[str, Mapping[str, object]]]

{enums_block}{configuration_types}

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


def _python_configuration_types(plugin: Plugin) -> str:
    if plugin.configuration is None:
        return "Configuration = Mapping[str, object]"
    declarations: list[str] = []

    def annotation(item: ConfigurationProperty, path: tuple[str, ...]) -> str:
        name = "".join(part[:1].upper() + part[1:] for part in (*path, item.name))
        if item.enum:
            alias = name + "Value"
            declarations.append(
                f"{alias} = Literal[{', '.join(json.dumps(value) for value in item.enum)}]"
            )
            return alias
        if item.kind == "object":
            return name
        if item.kind == "array":
            assert item.items is not None
            return f"list[{annotation(item.items, (*path, item.name))}]"
        return {
            "boolean": "bool",
            "integer": "int",
            "number": "float | int",
            "string": "str",
        }[item.kind]

    def typed_dict(
        name: str, properties: tuple[ConfigurationProperty, ...], path: tuple[str, ...]
    ) -> None:
        fields: list[tuple[ConfigurationProperty, str]] = []
        for item in properties:
            field_annotation = annotation(item, path)
            fields.append((item, field_annotation))
            if item.kind == "object":
                child_name = "".join(
                    part[:1].upper() + part[1:] for part in (*path, item.name)
                )
                typed_dict(child_name, item.properties, (*path, item.name))
        required = [(item, value) for item, value in fields if item.required]
        optional = [(item, value) for item, value in fields if not item.required]
        if required and optional:
            base = name + "Required"
            declarations.append(
                "class "
                + base
                + "(TypedDict):\n"
                + "\n".join(f"    {item.name}: {value}" for item, value in required)
            )
            declarations.append(
                "class "
                + name
                + "("
                + base
                + ", total=False):\n"
                + "\n".join(f"    {item.name}: {value}" for item, value in optional)
            )
        else:
            total = "" if required else ", total=False"
            declarations.append(
                "class "
                + name
                + f"(TypedDict{total}):\n"
                + "\n".join(f"    {item.name}: {value}" for item, value in fields)
            )

    typed_dict("Configuration", plugin.configuration.properties, ())
    return "\n\n".join(declarations)


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

    def declaration(out_annotation: str, result: str) -> str:
        parameters = [*arguments, f"*, out: {out_annotation}"]
        single = f"def {operation.name}({', '.join(parameters)}) -> {result}: ..."
        if len(single) <= 88:
            return single
        inside = ", ".join(parameters)
        if len(inside) <= 84:
            return f"def {operation.name}(\n    {inside}\n) -> {result}: ..."
        rendered = "\n".join(f"    {item}," for item in arguments)
        return (
            f"def {operation.name}(\n{rendered}\n    *,\n"
            f"    out: {out_annotation},\n) -> {result}: ..."
        )

    return "\n".join(
        (
            "@overload",
            declaration("None = None", output_type),
            "@overload",
            declaration(out_type, out_type),
        )
    )


def _python_name(name: str) -> str:
    return "".join(
        ("_" + character.lower()) if character.isupper() else character
        for character in name
    )


def _worker_python_adapter(plugin: Plugin) -> str:
    configuration_fingerprint = _configuration_fingerprint(plugin)
    configuration_fingerprint_literal = (
        "(\n    " + json.dumps(configuration_fingerprint) + "\n)"
        if configuration_fingerprint
        else "None"
    )
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
        f"            {json.dumps(item.name)}: _adapt_{item.name}(implementations[{json.dumps(item.name)}]),"
        for item in plugin.operations
    )
    interface_fingerprint = hashlib.sha256(
        json.dumps(
            _interface_document(plugin),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    declarations = json.dumps(
        [asdict(item) for item in plugin.operations],
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"""# Generated by wmfs-tool. Do not edit.
import json
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
INTERFACE_FINGERPRINT_SHA256 = bytes.fromhex(
    {json.dumps(interface_fingerprint)}
)
OPERATION_COUNT = {len(plugin.operations)}
STARTUP_CAPABILITIES = {_startup_capabilities(plugin)}
HAS_INITIALIZE = {plugin.lifecycle.initialize!r}
HAS_SHUTDOWN = {plugin.lifecycle.shutdown!r}
CONFIGURATION_SCHEMA_VERSION = {plugin.configuration.schema_version if plugin.configuration else 0}
CONFIGURATION_FINGERPRINT = {configuration_fingerprint_literal}
CONFIGURATION_FINGERPRINT_SHA256 = (
    bytes.fromhex(CONFIGURATION_FINGERPRINT.removeprefix("sha256:"))
    if CONFIGURATION_FINGERPRINT
    else bytes(32)
)
WORKER_DECLARATIONS = json.loads(
    {declarations!r}
)


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


class BoundOperations(dict[str, OperationHandler]):
    plugin_name = PLUGIN_NAME
    plugin_version = PLUGIN_VERSION
    protocol_version = PROTOCOL_VERSION
    metadata_fingerprint = METADATA_FINGERPRINT
    interface_fingerprint = INTERFACE_FINGERPRINT_SHA256
    configuration_schema_version = CONFIGURATION_SCHEMA_VERSION
    configuration_fingerprint = CONFIGURATION_FINGERPRINT_SHA256
    operation_count = OPERATION_COUNT
    startup_capabilities = STARTUP_CAPABILITIES
    declarations = WORKER_DECLARATIONS

    def __init__(self, *args, initialize=None, shutdown=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize = initialize
        self.shutdown = shutdown


{adapters}


_OPERATION_NAMES = frozenset(
    {{
{names}
    }}
)


def bind_plugin(
    implementations: Mapping[str, Implementation],
    *,
    initialize=None,
    shutdown=None,
) -> BoundOperations:
    if HAS_INITIALIZE != callable(initialize):
        raise ValueError("initialize hook does not match generated lifecycle")
    if HAS_SHUTDOWN != callable(shutdown):
        raise ValueError("shutdown hook does not match generated lifecycle")
    if set(implementations) != _OPERATION_NAMES:
        missing = sorted(_OPERATION_NAMES - set(implementations))
        unknown = sorted(set(implementations) - _OPERATION_NAMES)
        raise ValueError(
            f"Implementations do not match generated metadata: "
            f"missing={{missing}}, unknown={{unknown}}"
        )
    return BoundOperations(
        {{
{bindings}
        }},
        initialize=initialize,
        shutdown=shutdown,
    )


def bind_operations(
    implementations: Mapping[str, Implementation],
) -> BoundOperations:
    if set(implementations) != _OPERATION_NAMES:
        missing = sorted(_OPERATION_NAMES - set(implementations))
        unknown = sorted(set(implementations) - _OPERATION_NAMES)
        raise ValueError(
            f"Implementations do not match generated metadata: "
            f"missing={{missing}}, unknown={{unknown}}"
        )
    return BoundOperations(
        {{
{bindings}
        }}
    )
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
