from wmfs_plugin.schema import (
    PROTOCOL_VERSION,
    load_runtime_schema,
    load_tensor_schema,
    schema_root,
)


def test_protocol_schemas_are_packaged_and_loadable() -> None:
    root = schema_root()

    assert (root / "wmfs" / "runtime.capnp").is_file()
    assert (root / "wmfs" / "tensor.capnp").is_file()
    assert PROTOCOL_VERSION == 8
    assert int(load_runtime_schema().protocolVersion) == PROTOCOL_VERSION
    assert load_tensor_schema().TensorDescriptor is not None
