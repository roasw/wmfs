#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>

#include "wmfs/tensor.capnp.h"

namespace wmfs::reference {

struct MappingSpec {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    std::uint64_t allocation_id;
    std::uint64_t byte_length;
    std::uint64_t invocation_id;
    bool writable;
    bool arena;
};

class TensorLease {
  public:
    TensorLease(at::Tensor tensor, std::shared_ptr<void> mapping);

    [[nodiscard]] const at::Tensor &tensor() const noexcept;
    [[nodiscard]] at::Tensor &tensor() noexcept;

  private:
    std::shared_ptr<void> mapping_;
    at::Tensor tensor_;
};

class MappedBufferCache {
  public:
    MappedBufferCache();
    ~MappedBufferCache();

    MappedBufferCache(const MappedBufferCache &) = delete;
    MappedBufferCache &operator=(const MappedBufferCache &) = delete;

    void map(MappingSpec spec, int fd);
    void retire(std::uint64_t buffer_id, std::uint32_t generation,
                std::uint64_t allocation_id);
    [[nodiscard]] TensorLease tensor(TensorDescriptor::Reader descriptor,
                                     std::uint64_t invocation_id,
                                     bool require_writable = false);
    void finish_invocation(std::uint64_t invocation_id);
    void close();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

void receive_buffer_transfers(int control_fd, MappedBufferCache &cache);

} // namespace wmfs::reference
