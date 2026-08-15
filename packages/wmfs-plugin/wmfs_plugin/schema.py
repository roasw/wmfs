from functools import cache
from pathlib import Path
from types import ModuleType

import capnp

_SCHEMA_ROOT = Path(__file__).parent / "schemas"


def schema_root() -> Path:
    return _SCHEMA_ROOT


@cache
def load_runtime_schema() -> ModuleType:
    return capnp.load(
        str(_SCHEMA_ROOT / "wmfs" / "runtime.capnp"),
        imports=[str(_SCHEMA_ROOT)],
    )


@cache
def load_tensor_schema() -> ModuleType:
    return capnp.load(
        str(_SCHEMA_ROOT / "wmfs" / "tensor.capnp"),
        imports=[str(_SCHEMA_ROOT)],
    )


PROTOCOL_VERSION = int(load_runtime_schema().protocolVersion)
