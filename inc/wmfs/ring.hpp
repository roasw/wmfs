#pragma once

#include "wmfs/protocol/ring.h"
#include "wmfs/unique_fd.hpp"

#include <chrono>
#include <cstdint>
#include <memory>

namespace wmfs {

enum class RingWaitResult { success, timeout, closed };

using RingDeadline = std::chrono::steady_clock::time_point;

class RingProducer;
class RingConsumer;

/// Owns a newly-created shared ring and its notification descriptors.
class RingOwner {
  public:
    static RingOwner create(std::uint32_t capacity,
                            std::uint64_t session_generation);

    ~RingOwner();
    RingOwner(RingOwner &&) noexcept;
    RingOwner &operator=(RingOwner &&) noexcept;
    RingOwner(const RingOwner &) = delete;
    RingOwner &operator=(const RingOwner &) = delete;

    [[nodiscard]] int ring_fd() const noexcept;
    [[nodiscard]] int data_event_fd() const noexcept;
    [[nodiscard]] int space_event_fd() const noexcept;
    [[nodiscard]] UniqueFd duplicate_ring_fd() const;
    [[nodiscard]] UniqueFd duplicate_data_event_fd() const;
    [[nodiscard]] UniqueFd duplicate_space_event_fd() const;

    [[nodiscard]] RingProducer producer() const;
    [[nodiscard]] RingConsumer consumer() const;
    void close();

  private:
    struct Impl;
    explicit RingOwner(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

/// The sole writer endpoint for one SPSC ring.
class RingProducer {
  public:
    static RingProducer borrow(UniqueFd ring_fd, UniqueFd data_event_fd,
                               UniqueFd space_event_fd,
                               std::uint64_t expected_generation);

    ~RingProducer();
    RingProducer(RingProducer &&) noexcept;
    RingProducer &operator=(RingProducer &&) noexcept;
    RingProducer(const RingProducer &) = delete;
    RingProducer &operator=(const RingProducer &) = delete;

    [[nodiscard]] bool try_push(const wmfs_ring_record_v1 &record);
    [[nodiscard]] RingWaitResult
    push(const wmfs_ring_record_v1 &record,
         RingDeadline deadline = RingDeadline::max());
    void close();

    [[nodiscard]] int ring_fd() const noexcept;
    [[nodiscard]] int data_event_fd() const noexcept;
    [[nodiscard]] int space_event_fd() const noexcept;
    [[nodiscard]] UniqueFd duplicate_ring_fd() const;
    [[nodiscard]] UniqueFd duplicate_data_event_fd() const;
    [[nodiscard]] UniqueFd duplicate_space_event_fd() const;

  private:
    friend class RingOwner;
    struct Impl;
    explicit RingProducer(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

/// The sole reader endpoint for one SPSC ring.
class RingConsumer {
  public:
    static RingConsumer borrow(UniqueFd ring_fd, UniqueFd data_event_fd,
                               UniqueFd space_event_fd,
                               std::uint64_t expected_generation);

    ~RingConsumer();
    RingConsumer(RingConsumer &&) noexcept;
    RingConsumer &operator=(RingConsumer &&) noexcept;
    RingConsumer(const RingConsumer &) = delete;
    RingConsumer &operator=(const RingConsumer &) = delete;

    [[nodiscard]] bool try_pop(wmfs_ring_record_v1 &record);
    [[nodiscard]] RingWaitResult
    pop(wmfs_ring_record_v1 &record,
        RingDeadline deadline = RingDeadline::max());
    void close();

    [[nodiscard]] int ring_fd() const noexcept;
    [[nodiscard]] int data_event_fd() const noexcept;
    [[nodiscard]] int space_event_fd() const noexcept;
    [[nodiscard]] UniqueFd duplicate_ring_fd() const;
    [[nodiscard]] UniqueFd duplicate_data_event_fd() const;
    [[nodiscard]] UniqueFd duplicate_space_event_fd() const;

  private:
    friend class RingOwner;
    struct Impl;
    explicit RingConsumer(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};

} // namespace wmfs
