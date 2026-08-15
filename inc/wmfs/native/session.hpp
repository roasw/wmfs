#pragma once

#include <cstdint>
#include <memory>
#include <string>
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

struct TensorDescriptor {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    std::uint64_t allocation_id;
    std::uint64_t offset;
    std::uint64_t byte_length;
    std::string dtype;
    std::vector<std::uint64_t> shape;
    std::vector<std::int64_t> strides;
};

enum class ScalarKind { boolean, float64, int64, text };

struct ScalarArgument {
    std::uint16_t parameter;
    ScalarKind kind;
    bool boolean_value{};
    double float64_value{};
    std::int64_t int64_value{};
    std::string text_value;
};

class Session {
  public:
    Session(int rpc_fd, int control_fd, std::uint64_t expected_fingerprint);
    ~Session();

    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;

    bool mapping_required(const Mapping &mapping) const;
    void map_buffer(const Mapping &mapping, int fd);
    void retire_buffer(const Mapping &mapping);
    void invoke(std::uint64_t invocation_id, std::uint32_t operation_id,
                const std::vector<TensorDescriptor> &inputs,
                const std::vector<TensorDescriptor> &outputs,
                const std::vector<ScalarArgument> &scalars);
    void ping(std::uint64_t nonce);
    void close();

    std::uint64_t transfer_count() const;
    std::uint64_t retirement_count() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace wmfs::native
