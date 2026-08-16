#pragma once

#include <ATen/core/Tensor.h>

#include <tuple>

namespace wmfs::reference {

at::Tensor matmul(const at::Tensor &a, const at::Tensor &b);
at::Tensor &matmul_out(const at::Tensor &a, const at::Tensor &b,
                       at::Tensor &out);

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd(const at::Tensor &a,
                                                   bool full_matrices);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
svd_out(const at::Tensor &a, bool full_matrices, at::Tensor &u, at::Tensor &s,
        at::Tensor &vh);

at::Tensor add_scalar(const at::Tensor &a, double value);
at::Tensor &add_scalar_out(const at::Tensor &a, double value, at::Tensor &out);

std::tuple<at::Tensor, at::Tensor>
matmul_vjp_out(const at::Tensor &a, const at::Tensor &b,
               const at::Tensor &result_cotangent, at::Tensor &a_gradient,
               at::Tensor &b_gradient);

at::Tensor &add_scalar_vjp_out(const at::Tensor &result_cotangent,
                               at::Tensor &a_gradient);

} // namespace wmfs::reference
