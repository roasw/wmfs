from typing import Final, Mapping, NamedTuple, overload

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

Configuration = Mapping[str, object]

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

@overload
def scale(a: torch.Tensor, factor: float, *, out: None = None) -> torch.Tensor: ...
@overload
def scale(a: torch.Tensor, factor: float, *, out: torch.Tensor) -> torch.Tensor: ...
