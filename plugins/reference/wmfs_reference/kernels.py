from __future__ import annotations

import threading
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wmfs_plugin import Logger

_state_lock = threading.Lock()
_state: Mapping[str, object] | None = None
_state_references = 0


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def initialize(config: dict[str, object], logger: Logger) -> None:
    global _state, _state_references
    frozen = _freeze(config)
    assert isinstance(frozen, Mapping)
    with _state_lock:
        if _state is not None:
            if _state == frozen:
                _state_references += 1
                return
            raise RuntimeError("reference plugin is already initialized")
        _state = frozen
        _state_references = 1
    if config.get("emit_diagnostics") and logger.enabled(20):
        logger.info(
            "reference plugin initialized",
            fields={"threads": int(config.get("threads", 1))},
        )


def shutdown() -> None:
    global _state, _state_references
    with _state_lock:
        if _state_references:
            _state_references -= 1
        if not _state_references:
            _state = None


def matmul(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.matmul(a, b, out=out)


def svd(
    a: torch.Tensor,
    full_matrices: bool = True,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(a, full_matrices=full_matrices, out=out)


def add_scalar(
    a: torch.Tensor, value: float, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("value must be numeric")
    return torch.add(a, float(value), out=out)


def nonzero(
    a: torch.Tensor, order: int | str = 0, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    result = torch.nonzero(a)
    if order in (1, "columnMajor"):
        result = result.flip(1)
    elif order not in (0, "rowMajor"):
        raise ValueError("Scalar 'order' is outside enum 'IndexOrder'")
    return result if out is None else out.copy_(result)


def matmul_vjp(
    a: torch.Tensor,
    b: torch.Tensor,
    result_cotangent: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.matmul(result_cotangent, b.mT, out=out[0])
    torch.matmul(a.mT, result_cotangent, out=out[1])
    return out


def add_scalar_vjp(
    result_cotangent: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    return out.copy_(result_cotangent)
