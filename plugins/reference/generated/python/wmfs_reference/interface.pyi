from typing import Final, Mapping, NamedTuple

PLUGIN_NAME: Final[str]
API_NAMESPACE: Final[str]
PLUGIN_VERSION: Final[str]
FORMAT_VERSION: Final[int]
ABI_VERSION: Final[int]
PROTOCOL_VERSION: Final[int]
INTERFACE_FINGERPRINT: Final[str]

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
