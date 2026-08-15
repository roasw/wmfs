#include "wmfs/reference/kernels.hpp"

#include <torch/library.h>

TORCH_LIBRARY(wmfs_reference, module) {
    module.def("matmul(Tensor a, Tensor b) -> Tensor");
    module.def(
        "matmul.out(Tensor a, Tensor b, *, Tensor(a!) out) -> Tensor(a!)");
    module.def(
        "svd(Tensor a, bool full_matrices=True) -> (Tensor, Tensor, Tensor)");
    module.def("svd.out(Tensor a, bool full_matrices=True, *, Tensor(a!) u, "
               "Tensor(b!) s, Tensor(c!) vh) -> (Tensor(a!), Tensor(b!), "
               "Tensor(c!))");
    module.def("add_scalar(Tensor a, float value) -> Tensor");
    module.def("add_scalar.out(Tensor a, float value, *, Tensor(a!) out) -> "
               "Tensor(a!)");
}

TORCH_LIBRARY_IMPL(wmfs_reference, CompositeImplicitAutograd, module) {
    module.impl("matmul", TORCH_FN(wmfs::reference::matmul));
    module.impl("svd", TORCH_FN(wmfs::reference::svd));
    module.impl("add_scalar", TORCH_FN(wmfs::reference::add_scalar));
}

TORCH_LIBRARY_IMPL(wmfs_reference, CPU, module) {
    module.impl("matmul.out", TORCH_FN(wmfs::reference::matmul_out));
    module.impl("svd.out", TORCH_FN(wmfs::reference::svd_out));
    module.impl("add_scalar.out", TORCH_FN(wmfs::reference::add_scalar_out));
}
