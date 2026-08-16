#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wmfs::native {

struct Mapping {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    std::uint64_t allocation_id;
    std::uint64_t byte_length;
    bool writable;
    bool arena;
    std::uint64_t invocation_id;
};

enum class TensorDType { float32, float64, int64, uint8 };

struct TensorDescriptor {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    std::uint64_t allocation_id;
    std::uint64_t offset;
    std::uint64_t byte_length;
    TensorDType dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::int64_t> strides;
};

using TensorDescriptors = std::vector<const TensorDescriptor *>;

enum class ScalarKind { boolean, float64, int64, text };

struct ScalarArgument {
    std::uint16_t parameter;
    ScalarKind kind;
    bool boolean_value{};
    double float64_value{};
    std::int64_t int64_value{};
    std::string text_value;
};

struct InvocationProfile {
    std::uint64_t queue_wait_ns{};
    std::uint64_t rpc_ns{};
    std::uint64_t worker_input_views_ns{};
    std::uint64_t worker_output_views_ns{};
    std::uint64_t worker_dispatch_ns{};
    std::uint64_t worker_kernel_ns{};
};

class Session {
  public:
    Session(int rpc_fd, int control_fd, std::uint64_t expected_fingerprint,
            double startup_timeout_seconds, double request_timeout_seconds,
            double fd_transfer_timeout_seconds);
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    bool mapping_required(const Mapping &mapping) const;
    void map_buffer(const Mapping &mapping, int fd);
    std::vector<bool>
    map_buffers(std::vector<std::pair<Mapping, int>> mappings);
    void retire_buffer(const Mapping &mapping);
    void retire_buffers(const std::vector<Mapping> &mappings);
    void abort_invocation(std::uint64_t invocation_id);
    void invoke(std::uint64_t invocation_id, std::uint32_t operation_id,
                const TensorDescriptors &inputs,
                const TensorDescriptors &outputs,
                const std::vector<ScalarArgument> &scalars);
    InvocationProfile
    invoke_profiled(std::uint64_t invocation_id, std::uint32_t operation_id,
                    const TensorDescriptors &inputs,
                    const TensorDescriptors &outputs,
                    const std::vector<ScalarArgument> &scalars);
    void ping(std::uint64_t nonce);
    std::vector<std::uint8_t> metadata();
    std::vector<std::uint8_t> environment();
    void close();

    std::uint64_t transfer_count() const;
    std::uint64_t mapping_batch_count() const;
    std::uint64_t retirement_count() const;
    std::uint64_t retirement_batch_count() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace wmfs::native
