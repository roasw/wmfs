#pragma once

#include <ATen/core/Tensor.h>

#include <tuple>

namespace wmfs::reference {

/// @brief Compute a matrix product with an allocated result.
at::Tensor matmul(const at::Tensor &a, const at::Tensor &b);
/// @brief Compute a matrix product into runtime-owned storage.
at::Tensor &matmul_out(const at::Tensor &a, const at::Tensor &b,
                       at::Tensor &out);

/// @brief Compute a singular value decomposition with allocated outputs.
std::tuple<at::Tensor, at::Tensor, at::Tensor> svd(const at::Tensor &a,
                                                   bool full_matrices);
/// @brief Compute a singular value decomposition into runtime-owned outputs.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
svd_out(const at::Tensor &a, bool full_matrices, at::Tensor &u, at::Tensor &s,
        at::Tensor &vh);

/// @brief Add a scalar with an allocated result.
at::Tensor add_scalar(const at::Tensor &a, double value);
/// @brief Add a scalar into runtime-owned storage.
at::Tensor &add_scalar_out(const at::Tensor &a, double value, at::Tensor &out);

/// @brief Return nonzero indices in runtime-owned dynamic storage.
at::Tensor &nonzero_out(const at::Tensor &a, at::Tensor &out);

/// @brief Compute the first-order vector-Jacobian product for matrix multiply.
std::tuple<at::Tensor, at::Tensor>
matmul_vjp_out(const at::Tensor &a, const at::Tensor &b,
               const at::Tensor &result_cotangent, at::Tensor &a_gradient,
               at::Tensor &b_gradient);

/// @brief Copy the add-scalar result cotangent into its input gradient.
at::Tensor &add_scalar_vjp_out(const at::Tensor &result_cotangent,
                               at::Tensor &a_gradient);

} // namespace wmfs::reference
