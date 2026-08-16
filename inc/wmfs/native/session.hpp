#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wmfs::native {

/// @brief Metadata required to map one runtime-owned shared buffer.
struct Mapping {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    std::uint64_t allocation_id;
    std::uint64_t byte_length;
    bool writable;
    bool arena;
    std::uint64_t invocation_id;
};

/// @brief Tensor element types supported by the native control path.
enum class TensorDType { float32, float64, int64, uint8 };

/// @brief Process-independent tensor view metadata.
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

/// @brief Scalar argument types represented by the invocation protocol.
enum class ScalarKind { boolean, float64, int64, text };

/// @brief One indexed scalar argument for an operation invocation.
struct ScalarArgument {
    std::uint16_t parameter;
    ScalarKind kind;
    bool boolean_value{};
    double float64_value{};
    std::int64_t int64_value{};
    std::string text_value;
};

/// @brief Timing values returned by a profiled native invocation.
struct InvocationProfile {
    std::uint64_t queue_wait_ns{};
    std::uint64_t rpc_ns{};
    std::uint64_t worker_input_views_ns{};
    std::uint64_t worker_output_views_ns{};
    std::uint64_t worker_dispatch_ns{};
    std::uint64_t worker_kernel_ns{};
};

/// @brief Synchronous native client for worker RPC and FD-control traffic.
///
/// Session serializes KJ RPC work on its implementation thread. The Python
/// runtime owns process lifecycle and must call close before releasing the
/// worker process.
class Session {
  public:
    /// @brief Adopt connected RPC/control sockets and validate the worker.
    Session(int rpc_fd, int control_fd, std::uint64_t expected_fingerprint,
            double startup_timeout_seconds, double request_timeout_seconds,
            double fd_transfer_timeout_seconds);
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    /// @brief Return whether the worker needs this mapping generation.
    bool mapping_required(const Mapping &mapping) const;
    /// @brief Transfer and map one buffer.
    void map_buffer(const Mapping &mapping, int fd);
    /// @brief Transfer multiple buffers in one acknowledged control batch.
    std::vector<bool>
    map_buffers(std::vector<std::pair<Mapping, int>> mappings);
    /// @brief Retire one worker mapping.
    void retire_buffer(const Mapping &mapping);
    /// @brief Retire multiple mappings in one acknowledged control batch.
    void retire_buffers(const std::vector<Mapping> &mappings);
    /// @brief Release invocation-scoped mappings after pre-dispatch failure.
    void abort_invocation(std::uint64_t invocation_id);
    /// @brief Invoke an operation and wait for completion.
    void invoke(std::uint64_t invocation_id, std::uint32_t operation_id,
                const TensorDescriptors &inputs,
                const TensorDescriptors &outputs,
                const std::vector<ScalarArgument> &scalars);
    /// @brief Invoke an operation and return native/worker timing metrics.
    InvocationProfile
    invoke_profiled(std::uint64_t invocation_id, std::uint32_t operation_id,
                    const TensorDescriptors &inputs,
                    const TensorDescriptors &outputs,
                    const std::vector<ScalarArgument> &scalars);
    /// @brief Verify RPC responsiveness with a nonce round trip.
    void ping(std::uint64_t nonce);
    /// @brief Return serialized plugin metadata obtained during startup.
    std::vector<std::uint8_t> metadata();
    /// @brief Return serialized worker environment metadata.
    std::vector<std::uint8_t> environment();
    /// @brief Close RPC/control resources. Safe to call repeatedly.
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
