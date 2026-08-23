from enum import IntEnum
from typing import Final, Literal, Mapping, NamedTuple, TypedDict, overload

import torch

PLUGIN_NAME: Final[str]
API_NAMESPACE: Final[str]
PLUGIN_VERSION: Final[str]
FORMAT_VERSION: Final[int]
ABI_VERSION: Final[int]
PROTOCOL_VERSION: Final[int]
HAS_INITIALIZE: Final[bool]
HAS_SHUTDOWN: Final[bool]
INTERFACE_FINGERPRINT: Final[str]
CONFIGURATION_SCHEMA_VERSION: Final[int | None]
CONFIGURATION_FINGERPRINT: Final[str | None]
CONFIGURATION_SCHEMA: Final[Mapping[str, object] | None]
CONFIGURATION_EXAMPLES: Final[Mapping[str, Mapping[str, object]]]

class IndexOrder(IntEnum):
    rowMajor: Final[IndexOrder]
    columnMajor: Final[IndexOrder]

PrecisionValue = Literal["fast", "balanced", "accurate"]

SolverAlgorithmValue = Literal["divideAndConquer", "qrIteration"]

class SolverRequired(TypedDict):
    algorithm: SolverAlgorithmValue

class Solver(SolverRequired, total=False):
    tolerance: float | int

class Configuration(TypedDict, total=False):
    threads: int
    precision: PrecisionValue
    emit_diagnostics: bool
    tags: list[str]
    solver: Solver

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
def svd(
    a: torch.Tensor, full_matrices: bool = True, *, out: None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@overload
def svd(
    a: torch.Tensor,
    full_matrices: bool = True,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
@overload
def add_scalar(a: torch.Tensor, value: float, *, out: None = None) -> torch.Tensor: ...
@overload
def add_scalar(a: torch.Tensor, value: float, *, out: torch.Tensor) -> torch.Tensor: ...
@overload
def nonzero(
    a: torch.Tensor, order: IndexOrder = IndexOrder.rowMajor, *, out: None = None
) -> torch.Tensor: ...
@overload
def nonzero(
    a: torch.Tensor, order: IndexOrder = IndexOrder.rowMajor, *, out: torch.Tensor
) -> torch.Tensor: ...
