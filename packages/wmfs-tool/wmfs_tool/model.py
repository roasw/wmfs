from dataclasses import dataclass
from typing import TypeAlias

ScalarDefault: TypeAlias = bool | float | int | str | None


@dataclass(frozen=True)
class TensorParameter:
    name: str
    access: str = "read_only"
    dtype_variable: str | None = None
    dtypes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DTypeVariable:
    name: str
    dtypes: tuple[str, ...]


@dataclass(frozen=True)
class Enum:
    name: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class ScalarParameter:
    name: str
    kind: str
    required: bool
    default: ScalarDefault
    enum: str | None = None
    enum_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class Dimension:
    kind: str
    input: int | None = None
    axis: int | None = None
    scalar: int | None = None
    operands: tuple["Dimension", ...] = ()
    when_true: "Dimension | None" = None
    when_false: "Dimension | None" = None


@dataclass(frozen=True)
class DType:
    kind: str
    input: int | None = None
    scalar: int | None = None
    value: str | None = None
    variable: str | None = None


@dataclass(frozen=True)
class Output:
    name: str
    allocation: str
    dimensions: tuple[Dimension, ...]
    same_shape_as_input: int | None
    dtype: DType | None


@dataclass(frozen=True)
class Vjp:
    operation_id: int
    saved_inputs: tuple[int, ...]
    saved_outputs: tuple[int, ...]
    output_cotangents: tuple[int, ...]
    input_gradients: tuple[int, ...]
    scalar_parameters: tuple[int, ...]


@dataclass(frozen=True)
class Operation:
    operation_id: int
    name: str
    internal: bool
    inputs: tuple[TensorParameter, ...]
    outputs: tuple[Output, ...]
    scalars: tuple[ScalarParameter, ...]
    vjp: Vjp | None
    dtype_variables: tuple[DTypeVariable, ...] = ()


@dataclass(frozen=True)
class Plugin:
    format_version: int
    abi_version: int
    protocol_version: int
    name: str
    version: str
    namespace: str
    python_module: str
    worker: str
    schema: str
    interface: str
    deployment_root: str
    operations: tuple[Operation, ...]
    enums: tuple[Enum, ...] = ()
