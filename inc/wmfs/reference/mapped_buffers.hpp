#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>

#include "wmfs/tensor.capnp.h"

namespace wmfs::reference {

/// @brief Worker-side mapping metadata received from the runtime.
struct MappingSpec {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    std::uint64_t allocation_id;
    std::uint64_t byte_length;
    std::uint64_t invocation_id;
    bool writable;
    bool arena;
};

/// @brief Invocation-local ATen tensor handle.
///
/// The tensor's ATen storage independently retains its mapped region, so
/// aliases remain valid after this lease and the mapping cache entry are
/// released.
class TensorLease {
  public:
    /// @brief Take ownership of an ATen tensor view.
    explicit TensorLease(at::Tensor tensor);

    [[nodiscard]] const at::Tensor &tensor() const noexcept;
    [[nodiscard]] at::Tensor &tensor() noexcept;

  private:
    at::Tensor tensor_;
};

/// @brief Validate descriptors and construct zero-copy ATen views over
/// mappings.
///
/// Cache mutation is thread-safe. Retiring a cache entry does not unmap a
/// region while retained ATen storage aliases still reference it.
class MappedBufferCache {
  public:
    MappedBufferCache();
    ~MappedBufferCache();

    MappedBufferCache(const MappedBufferCache &) = delete;
    MappedBufferCache &operator=(const MappedBufferCache &) = delete;

    /// @brief Adopt and map a transferred descriptor according to `spec`.
    void map(MappingSpec spec, int fd);
    /// @brief Retire one exact buffer generation and logical allocation.
    void retire(std::uint64_t buffer_id, std::uint32_t generation,
                std::uint64_t allocation_id);
    /// @brief Resolve a descriptor into an invocation-scoped tensor view.
    [[nodiscard]] TensorLease tensor(TensorDescriptor::Reader descriptor,
                                     std::uint64_t invocation_id,
                                     bool require_writable = false);
    /// @brief Retire writable mappings owned by an invocation.
    void finish_invocation(std::uint64_t invocation_id);
    /// @brief Drop all cache ownership while preserving retained tensor
    /// aliases.
    void close();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// @brief Receive batched SCM_RIGHTS mapping and retirement control messages.
void receive_buffer_transfers(int control_fd, MappedBufferCache &cache);

} // namespace wmfs::reference
