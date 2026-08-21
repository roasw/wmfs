#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wmfs::native {

/// @brief Metadata required to map one runtime-owned shared buffer.
struct Mapping {
    std::uint64_t buffer_id;     ///< Runtime buffer capability identifier.
    std::uint32_t generation;    ///< Reuse generation of the backing region.
    std::uint64_t allocation_id; ///< Logical allocation within the region.
    std::uint64_t byte_length;   ///< Number of mapped bytes.
    bool writable;               ///< Whether the worker may modify the mapping.
    bool arena;                  ///< Whether the mapping belongs to an arena.
    std::uint64_t invocation_id; ///< Invocation owning a writable mapping.
};

/// @brief Tensor element types supported by the native control path.
enum class TensorDType { float32, float64, int64, uint8 };

/// @brief Process-independent tensor view metadata.
struct TensorDescriptor {
    std::uint64_t buffer_id;     ///< Runtime buffer capability identifier.
    std::uint32_t generation;    ///< Reuse generation of the backing region.
    std::uint64_t allocation_id; ///< Logical allocation within the region.
    std::uint64_t offset;        ///< Byte offset from the mapped region start.
    std::uint64_t byte_length;   ///< Number of bytes visible to the tensor.
    TensorDType dtype;           ///< Element type.
    std::vector<std::uint64_t> shape;  ///< Tensor dimensions.
    std::vector<std::int64_t> strides; ///< Element strides.
};

using TensorDescriptors = std::vector<const TensorDescriptor *>;

/// @brief Scalar argument types represented by the invocation protocol.
enum class ScalarKind { boolean, float64, int64, text };

/// @brief One indexed scalar argument for an operation invocation.
struct ScalarArgument {
    std::uint16_t parameter; ///< Scalar parameter index in operation metadata.
    ScalarKind kind;         ///< Active scalar representation.
    bool boolean_value{};    ///< Boolean value when kind is boolean.
    double float64_value{};  ///< Floating-point value when kind is float64.
    std::int64_t int64_value{}; ///< Integer value when kind is int64.
    std::string text_value;     ///< String value when kind is text.
};

/// @brief Recoverable outcome returned by an operation invocation.
struct InvocationOutcome {
    std::string error_type;    ///< Empty on success.
    std::string error_message; ///< Worker-provided operation error text.
};

/// @brief One worker-planned dynamic output specification.
struct PlannedOutput {
    std::uint16_t output;
    std::vector<std::uint64_t> shape;
    TensorDType dtype;
};

/// @brief Recoverable result of dynamic output planning.
struct OutputPlanningResult {
    InvocationOutcome outcome;
    std::vector<PlannedOutput> outputs;
};

/// @brief Timing values returned by a profiled native invocation.
struct InvocationProfile {
    InvocationOutcome outcome;              ///< Recoverable operation result.
    std::uint64_t queue_wait_ns{};          ///< Client-side serialization wait.
    std::uint64_t rpc_ns{};                 ///< End-to-end RPC duration.
    std::uint64_t worker_input_views_ns{};  ///< Input view construction time.
    std::uint64_t worker_output_views_ns{}; ///< Output view construction time.
    std::uint64_t worker_dispatch_ns{};     ///< Worker adapter dispatch time.
    std::uint64_t worker_kernel_ns{};       ///< Numerical kernel time.
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
    InvocationOutcome invoke(std::uint64_t invocation_id,
                             std::uint32_t operation_id,
                             const TensorDescriptors &inputs,
                             const TensorDescriptors &outputs,
                             const std::vector<ScalarArgument> &scalars);
    /// @brief Invoke an operation and return native/worker timing metrics.
    InvocationProfile
    invoke_profiled(std::uint64_t invocation_id, std::uint32_t operation_id,
                    const TensorDescriptors &inputs,
                    const TensorDescriptors &outputs,
                    const std::vector<ScalarArgument> &scalars);
    /// @brief Ask the worker to determine data-dependent output metadata.
    OutputPlanningResult
    plan_outputs(std::uint64_t invocation_id, std::uint32_t operation_id,
                 const TensorDescriptors &inputs,
                 const std::vector<ScalarArgument> &scalars);
    /// @brief Verify RPC responsiveness with a nonce round trip.
    void ping(std::uint64_t nonce);
    /// @brief Return serialized plugin metadata obtained during startup.
    std::vector<std::uint8_t> metadata();
    /// @brief Return serialized worker environment metadata.
    std::vector<std::uint8_t> environment();
    /// @brief Close RPC/control resources. Safe to call repeatedly.
    void close();

    /// @brief Return the number of transferred mapping descriptors.
    std::uint64_t transfer_count() const;
    /// @brief Return the number of acknowledged mapping batches.
    std::uint64_t mapping_batch_count() const;
    /// @brief Return the number of retired mapping descriptors.
    std::uint64_t retirement_count() const;
    /// @brief Return the number of acknowledged retirement batches.
    std::uint64_t retirement_batch_count() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace wmfs::native
