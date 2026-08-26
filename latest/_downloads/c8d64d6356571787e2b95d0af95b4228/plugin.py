import torch

from wmfs_plugin import InvocationContext, OutputSpec, PluginBinding
from wmfs_reference import kernels
from wmfs_reference._generated import bind_plugin


def _matmul(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor) -> None:
    kernels.matmul(a, b, out=result)


def _svd(
    a: torch.Tensor,
    full_matrices: bool,
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
) -> None:
    kernels.svd(a, full_matrices=bool(full_matrices), out=(u, s, vh))


def _add_scalar(a: torch.Tensor, value: float, result: torch.Tensor) -> None:
    kernels.add_scalar(a, value, out=result)


def _nonzero(a: torch.Tensor, order: int, indices: torch.Tensor) -> None:
    kernels.nonzero(a, order, out=indices)


def _matmul_vjp(
    a: torch.Tensor,
    b: torch.Tensor,
    result_cotangent: torch.Tensor,
    a_gradient: torch.Tensor,
    b_gradient: torch.Tensor,
) -> None:
    kernels.matmul_vjp(a, b, result_cotangent, out=(a_gradient, b_gradient))


def _add_scalar_vjp(result_cotangent: torch.Tensor, a_gradient: torch.Tensor) -> None:
    kernels.add_scalar_vjp(result_cotangent, out=a_gradient)


def _plan_nonzero(context: InvocationContext) -> dict[str, OutputSpec]:
    a = context.input("a")
    count = int(torch.count_nonzero(a))
    if count == 0:
        raise ValueError("nonzero does not yet support an empty result")
    return {"indices": OutputSpec((count, a.ndim), torch.int64)}


_DIRECT_OPERATIONS = {
    "matmul": kernels.matmul,
    "svd": kernels.svd,
    "add_scalar": kernels.add_scalar,
    "matmul_vjp": kernels.matmul_vjp,
    "add_scalar_vjp": kernels.add_scalar_vjp,
    "nonzero": kernels.nonzero,
}

plugin = PluginBinding(
    bind_plugin(
        {
            "matmul": _matmul,
            "svd": _svd,
            "add_scalar": _add_scalar,
            "matmul_vjp": _matmul_vjp,
            "add_scalar_vjp": _add_scalar_vjp,
            "nonzero": _nonzero,
        },
        initialize=kernels.initialize,
        shutdown=kernels.shutdown,
    ),
    _DIRECT_OPERATIONS,
    output_planners={"nonzero": _plan_nonzero},
)
