from enum import IntEnum
from typing import Final, Mapping, NamedTuple, overload

import torch

PLUGIN_NAME: Final[str]
API_NAMESPACE: Final[str]
PLUGIN_VERSION: Final[str]
FORMAT_VERSION: Final[int]
ABI_VERSION: Final[int]
PROTOCOL_VERSION: Final[int]
INTERFACE_FINGERPRINT: Final[str]

class IndexOrder(IntEnum):
    rowMajor: Final[IndexOrder]
    columnMajor: Final[IndexOrder]

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
def matmul(a: torch.Tensor, b: torch.Tensor, *, out: None = None) -> torch.Tensor: ...
@overload
def matmul(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor: ...

@overload
def svd(a: torch.Tensor, full_matrices: bool = True, *, out: None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@overload
def svd(a: torch.Tensor, full_matrices: bool = True, *, out: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

@overload
def add_scalar(a: torch.Tensor, value: float, *, out: None = None) -> torch.Tensor: ...
@overload
def add_scalar(a: torch.Tensor, value: float, *, out: torch.Tensor) -> torch.Tensor: ...

@overload
def nonzero(a: torch.Tensor, order: IndexOrder = IndexOrder.rowMajor, *, out: None = None) -> torch.Tensor: ...
@overload
def nonzero(a: torch.Tensor, order: IndexOrder = IndexOrder.rowMajor, *, out: torch.Tensor) -> torch.Tensor: ...
