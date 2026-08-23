#include "wmfs/reference/kernels.hpp"

#include <ATen/ops/add.h>
#include <ATen/ops/linalg_svd.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/nonzero.h>
#include <ATen/ops/result_type.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace wmfs::reference {
namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void validate_output(const at::Tensor &output, at::IntArrayRef shape,
                     at::ScalarType dtype) {
    require(output.sizes().equals(shape) && output.scalar_type() == dtype,
            "Preallocated output has an invalid shape or dtype");
}

} // namespace

at::Tensor matmul(const at::Tensor &a, const at::Tensor &b) {
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul input dimensions are incompatible");
    return at::matmul(a, b);
}

at::Tensor &matmul_out(const at::Tensor &a, const at::Tensor &b,
                       at::Tensor &out) {
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul input dimensions are incompatible");
    std::array<std::int64_t, 2> shape{a.size(0), b.size(1)};
    validate_output(out, shape, a.scalar_type());
    return at::matmul_out(out, a, b);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd(const at::Tensor &a,
                                                   bool full_matrices) {
    require(a.dim() == 2, "svd initially supports two-dimensional tensors");
    return at::linalg_svd(a, full_matrices, std::nullopt);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
svd_out(const at::Tensor &a, bool full_matrices, at::Tensor &u, at::Tensor &s,
        at::Tensor &vh) {
    require(a.dim() == 2, "svd initially supports two-dimensional tensors");
    auto rows = a.size(0);
    auto columns = a.size(1);
    auto rank = std::min(rows, columns);
    std::array<std::int64_t, 2> u_shape{rows, full_matrices ? rows : rank};
    std::array<std::int64_t, 1> s_shape{rank};
    std::array<std::int64_t, 2> vh_shape{
        full_matrices ? columns : rank,
        columns,
    };
    validate_output(u, u_shape, a.scalar_type());
    validate_output(s, s_shape, a.scalar_type());
    validate_output(vh, vh_shape, a.scalar_type());
    at::linalg_svd_out(u, s, vh, a, full_matrices, std::nullopt);
    return {u, s, vh};
}

at::Tensor add_scalar(const at::Tensor &a, double value) {
    return at::add(a, at::Scalar(value), at::Scalar(1));
}

at::Tensor &add_scalar_out(const at::Tensor &a, double value, at::Tensor &out) {
    auto scalar = at::Scalar(value);
    validate_output(out, a.sizes(), at::result_type(a, scalar));
    return at::add_out(out, a, scalar, at::Scalar(1));
}

at::Tensor &nonzero_out(const at::Tensor &a, at::Tensor &out) {
    auto result = at::nonzero(a);
    validate_output(out, result.sizes(), at::ScalarType::Long);
    return out.copy_(result);
}

std::tuple<at::Tensor, at::Tensor>
matmul_vjp_out(const at::Tensor &a, const at::Tensor &b,
               const at::Tensor &result_cotangent, at::Tensor &a_gradient,
               at::Tensor &b_gradient) {
    require(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
            "matmul VJP input dimensions are incompatible");
    std::array<std::int64_t, 2> result_shape{a.size(0), b.size(1)};
    require(result_cotangent.sizes().equals(result_shape),
            "matmul VJP cotangent has an invalid shape");
    validate_output(a_gradient, a.sizes(), a.scalar_type());
    validate_output(b_gradient, b.sizes(), b.scalar_type());
    at::matmul_out(a_gradient, result_cotangent, b.transpose(0, 1));
    at::matmul_out(b_gradient, a.transpose(0, 1), result_cotangent);
    return {a_gradient, b_gradient};
}

at::Tensor &add_scalar_vjp_out(const at::Tensor &result_cotangent,
                               at::Tensor &a_gradient) {
    validate_output(a_gradient, result_cotangent.sizes(),
                    result_cotangent.scalar_type());
    return a_gradient.copy_(result_cotangent);
}

} // namespace wmfs::reference
