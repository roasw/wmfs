from collections.abc import Callable
from inspect import Parameter, Signature
from typing import Any

import torch

from wmfs.registry import OperationMetadata


def create_operation(
    runtime: Any,
    name: str,
    generation: int,
    metadata: OperationMetadata | None,
) -> Callable[..., object]:
    """Create a callable bound to one runtime operation-catalog generation."""
    signature = _operation_signature(metadata) if metadata is not None else None

    def operation(*args: object, **kwargs: object) -> object:
        if signature is None:
            out = kwargs.pop("out", None)
            return runtime.invoke_registered(name, generation, *args, out=out, **kwargs)

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        tensor_args = tuple(
            bound.arguments[python_parameter_name(parameter.name)]
            for parameter in metadata.tensor_inputs
        )
        scalar_kwargs = {
            python_parameter_name(parameter.name): bound.arguments[
                python_parameter_name(parameter.name)
            ]
            for parameter in metadata.scalar_parameters
        }
        return runtime.invoke_registered(
            name,
            generation,
            *tensor_args,
            out=bound.arguments["out"],
            **scalar_kwargs,
        )

    operation.__name__ = name.rsplit(".", 1)[-1]
    operation.__qualname__ = f"ops.{name}" if "." in name else name
    operation.__module__ = "wmfs"
    operation.__doc__ = f"Invoke the dynamically registered {name!r} operation."
    if signature is not None:
        operation.__signature__ = signature  # type: ignore[attr-defined]
    return operation


def python_parameter_name(name: str) -> str:
    """Convert a schema camelCase parameter to its Python spelling."""
    converted = []
    for character in name:
        if character.isupper():
            converted.extend(("_", character.lower()))
        else:
            converted.append(character)
    return "".join(converted)


def _operation_signature(metadata: OperationMetadata) -> Signature:
    parameters = [
        Parameter(
            python_parameter_name(parameter.name),
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=torch.Tensor,
        )
        for parameter in metadata.tensor_inputs
    ]
    scalar_annotations = {
        "boolean": bool,
        "float64": float,
        "int64": int,
        "text": str,
    }
    optional_seen = False
    keyword_only = False
    for parameter in metadata.scalar_parameters:
        if parameter.required and optional_seen:
            keyword_only = True
        parameters.append(
            Parameter(
                python_parameter_name(parameter.name),
                (
                    Parameter.KEYWORD_ONLY
                    if keyword_only
                    else Parameter.POSITIONAL_OR_KEYWORD
                ),
                default=(Parameter.empty if parameter.required else parameter.default),
                annotation=scalar_annotations[parameter.kind],
            )
        )
        optional_seen = optional_seen or not parameter.required
    parameters.append(
        Parameter("out", Parameter.KEYWORD_ONLY, default=None, annotation=object)
    )
    return Signature(parameters, return_annotation=object)
