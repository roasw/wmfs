#include "wmfs/native/session.hpp"
#include "wmfs/protocol/control.h"
#include "wmfs/ring.hpp"
#include "wmfs/unique_fd.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <fcntl.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace wmfs::native {
namespace {

struct MappingKey {
    std::uint64_t buffer_id;
    std::uint32_t generation;
    bool operator==(const MappingKey &) const = default;
};

struct MappingKeyHash {
    std::size_t operator()(const MappingKey &key) const noexcept {
        return std::hash<std::uint64_t>{}(key.buffer_id) ^
               (std::hash<std::uint32_t>{}(key.generation) << 1U);
    }
};

timeval timeout(double seconds) {
    if (!std::isfinite(seconds) || seconds <= 0)
        throw std::invalid_argument(
            "Native transport deadline must be positive");
    const auto integral = static_cast<time_t>(seconds);
    return timeval{integral,
                   static_cast<suseconds_t>((seconds - integral) * 1'000'000)};
}

void set_timeout(int fd, double seconds) {
    const auto flags = ::fcntl(fd, F_GETFL);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags & ~O_NONBLOCK) < 0)
        throw std::runtime_error(
            "Cannot configure blocking native control socket");
    const auto value = timeout(seconds);
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &value, sizeof(value)) < 0 ||
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &value, sizeof(value)) < 0)
        throw std::runtime_error("Cannot set native fixed-protocol deadline");
}

std::uint64_t monotonic_nanoseconds() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

std::uint64_t ordered_delta(std::uint64_t later, std::uint64_t earlier) {
    return later > earlier ? later - earlier : 0;
}

} // namespace

struct Session::Impl {
    struct PendingSubmission {
        std::mutex mutex;
        std::condition_variable ready;
        wmfs_ring_record_v1 completion{};
        std::exception_ptr error;
        std::uint32_t expected_kind{};
        std::uint32_t expected_flags{};
        std::uint64_t invocation_id{};
        std::uint32_t operation_id{};
        std::uint64_t completion_consumed_ns{};
        bool completed{};
    };

    static std::uint32_t completion_kind(std::uint32_t command_kind) {
        switch (command_kind) {
        case WMFS_RING_COMMAND_INVOKE:
            return WMFS_RING_COMPLETION_INVOKE;
        case WMFS_RING_COMMAND_PLAN_OUTPUTS:
            return WMFS_RING_COMPLETION_PLAN_OUTPUTS;
        case WMFS_RING_COMMAND_PING:
            return WMFS_RING_COMPLETION_PONG;
        case WMFS_RING_COMMAND_SHUTDOWN:
            return WMFS_RING_COMPLETION_SHUTDOWN;
        default:
            throw std::invalid_argument("Unknown native ring command kind");
        }
    }

    Impl(int lifecycle_fd, int fd_control, std::uint64_t,
         std::uint64_t ring_generation, std::uint32_t, double,
         double request_timeout, double fd_timeout, int command_ring_fd,
         int command_data_fd, int command_space_fd, int completion_ring_fd,
         int completion_data_fd, int completion_space_fd)
        : lifecycle(lifecycle_fd), control(fd_control),
          generation(ring_generation) {
        UniqueFd command_ring(command_ring_fd);
        UniqueFd command_data(command_data_fd);
        UniqueFd command_space(command_space_fd);
        UniqueFd completion_ring(completion_ring_fd);
        UniqueFd completion_data(completion_data_fd);
        UniqueFd completion_space(completion_space_fd);
        set_timeout(lifecycle.get(), request_timeout);
        set_timeout(control.get(), fd_timeout);
        const bool has_rings =
            command_ring_fd >= 0 || command_data_fd >= 0 ||
            command_space_fd >= 0 || completion_ring_fd >= 0 ||
            completion_data_fd >= 0 || completion_space_fd >= 0;
        if (has_rings) {
            if (command_ring_fd < 0 || command_data_fd < 0 ||
                command_space_fd < 0 || completion_ring_fd < 0 ||
                completion_data_fd < 0 || completion_space_fd < 0)
                throw std::invalid_argument(
                    "Native ring dispatcher requires all six descriptors");
            commands = std::make_unique<RingProducer>(RingProducer::borrow(
                std::move(command_ring), std::move(command_data),
                std::move(command_space), generation));
            completions = std::make_unique<RingConsumer>(RingConsumer::borrow(
                std::move(completion_ring), std::move(completion_data),
                std::move(completion_space), generation));
            completion_thread = std::thread([this] { consume_completions(); });
        }
    }

    ~Impl() {
        try {
            close();
        } catch (...) {
        }
    }

    void transfer(const std::vector<Mapping> &values,
                  const std::vector<bool> &maps, std::vector<int> fds) {
        if (values.empty() || values.size() > WMFS_CONTROL_MAX_FD_ENTRIES)
            throw std::invalid_argument("Native FD batch size is invalid");
        std::vector<wmfs_control_fd_entry_v1> entries(values.size());
        std::uint16_t fd_count = 0;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto &value = values[index];
            auto &entry = entries[index];
            entry.kind = maps[index] ? WMFS_CONTROL_FD_ENTRY_MAP
                                     : WMFS_CONTROL_FD_ENTRY_RETIRE;
            entry.flags =
                maps[index]
                    ? (value.writable ? WMFS_CONTROL_FD_FLAG_WRITABLE : 0) |
                          (value.arena ? WMFS_CONTROL_FD_FLAG_ARENA : 0)
                    : 0;
            entry.buffer_id = value.buffer_id;
            entry.generation = value.generation;
            entry.allocation_id = value.allocation_id;
            entry.invocation_id = maps[index] ? value.invocation_id : 0;
            entry.byte_length = value.byte_length;
            fd_count += maps[index];
        }
        const auto transfer_id = next_transfer++;
        const auto request_id = next_request++;
        wmfs_control_fd_batch_v1 batch{
            transfer_id,
            generation,
            static_cast<std::uint16_t>(values.size()),
            fd_count,
            WMFS_CONTROL_FD_BATCH_FLAG_TRANSACTIONAL,
            0};
        std::array<std::uint8_t, WMFS_CONTROL_MAX_PACKET_BYTES> packet{};
        wmfs_control_mutable_bytes_v1 output{packet.data(), packet.size(), 0};
        if (wmfs_control_encode_fd_batch_v1(request_id, &batch, entries.data(),
                                            &output) != 0)
            throw std::runtime_error("Cannot encode native FD batch");
        iovec vector{packet.data(), output.size};
        msghdr message{};
        message.msg_iov = &vector;
        message.msg_iovlen = 1;
        std::vector<std::byte> ancillary;
        if (!fds.empty()) {
            ancillary.resize(CMSG_SPACE(sizeof(int) * fds.size()));
            message.msg_control = ancillary.data();
            message.msg_controllen = ancillary.size();
            auto *item = CMSG_FIRSTHDR(&message);
            item->cmsg_level = SOL_SOCKET;
            item->cmsg_type = SCM_RIGHTS;
            item->cmsg_len = CMSG_LEN(sizeof(int) * fds.size());
            std::memcpy(CMSG_DATA(item), fds.data(), sizeof(int) * fds.size());
        }
        ssize_t sent;
        do {
            sent = ::sendmsg(control.get(), &message, MSG_NOSIGNAL);
        } while (sent < 0 && errno == EINTR);
        for (int fd : fds)
            ::close(fd);
        if (sent < 0 || static_cast<std::size_t>(sent) != output.size)
            throw std::runtime_error("Cannot send native FD batch");
        ssize_t received;
        do {
            received = ::recv(control.get(), packet.data(), packet.size(), 0);
        } while (received < 0 && errno == EINTR);
        if (received <= 0)
            throw std::runtime_error(
                "Native FD acknowledgement was not received");
        wmfs_control_fd_ack_v1 acknowledgement{};
        wmfs_control_bytes_v1 error{};
        std::uint64_t response_id = 0;
        const wmfs_control_bytes_v1 response{
            packet.data(), static_cast<std::size_t>(received)};
        if (wmfs_control_decode_fd_ack_v1(response, &response_id,
                                          &acknowledgement, &error) != 0 ||
            response_id != request_id ||
            acknowledgement.transfer_id != transfer_id ||
            acknowledgement.session_generation != generation)
            throw std::runtime_error("Worker rejected native FD batch");
        if (acknowledgement.status != WMFS_CONTROL_STATUS_OK)
            throw std::runtime_error(
                "Worker rejected native FD batch: " +
                std::string(reinterpret_cast<const char *>(error.data),
                            error.size));
    }

    void fail_ring(std::exception_ptr error) {
        std::vector<std::shared_ptr<PendingSubmission>> failed;
        {
            std::lock_guard lock(pending_mutex);
            if (!ring_error)
                ring_error = error;
            for (const auto &item : pending)
                failed.push_back(item.second);
            pending.clear();
        }
        for (const auto &submission : failed) {
            {
                std::lock_guard lock(submission->mutex);
                submission->error = ring_error;
                submission->completed = true;
            }
            submission->ready.notify_one();
        }
    }

    void consume_completions() {
        try {
            for (;;) {
                wmfs_ring_record_v1 completion{};
                const auto result = completions->pop(completion);
                if (result != RingWaitResult::success)
                    throw std::runtime_error(
                        result == RingWaitResult::timeout
                            ? "Native completion ring timed out"
                            : "Native completion ring closed");
                std::shared_ptr<PendingSubmission> submission;
                {
                    std::lock_guard lock(pending_mutex);
                    const auto item = pending.find(completion.submission_id);
                    if (item == pending.end())
                        throw std::runtime_error(
                            "Native completion has unknown submission ID");
                    submission = item->second;
                    if (completion.kind != submission->expected_kind ||
                        completion.flags != submission->expected_flags ||
                        completion.invocation_id != submission->invocation_id ||
                        completion.operation_id != submission->operation_id ||
                        completion.status < WMFS_RING_STATUS_OK ||
                        completion.status > WMFS_RING_STATUS_INTERNAL_ERROR ||
                        (completion.kind != WMFS_RING_COMPLETION_PLAN_OUTPUTS &&
                         completion.planned_output_count != 0) ||
                        completion.tensor_count != 0 ||
                        completion.scalar_count != 0)
                        throw std::runtime_error("Native completion identity "
                                                 "does not match command");
                    pending.erase(item);
                }
                {
                    std::lock_guard lock(submission->mutex);
                    submission->completion = completion;
                    if (completion.flags & WMFS_RING_RECORD_FLAG_PROFILE)
                        submission->completion_consumed_ns =
                            monotonic_nanoseconds();
                    submission->completed = true;
                }
                submission->ready.notify_one();
            }
        } catch (...) {
            fail_ring(std::current_exception());
        }
    }

    RingSubmissionResult submit_ring(wmfs_ring_record_v1 command,
                                     double timeout_seconds, bool profiled) {
        if (!commands || !completions)
            throw std::runtime_error("Native ring dispatcher is unavailable");
        if (!std::isfinite(timeout_seconds) || timeout_seconds <= 0)
            throw std::invalid_argument(
                "Native ring deadline must be positive");

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(timeout_seconds));
        const auto submitted_ns = profiled ? monotonic_nanoseconds() : 0;
        auto submission = std::make_shared<PendingSubmission>();
        submission->expected_kind = completion_kind(command.kind);
        submission->invocation_id = command.invocation_id;
        submission->operation_id = command.operation_id;
        {
            std::lock_guard lock(pending_mutex);
            if (ring_error)
                std::rethrow_exception(ring_error);
            if (next_submission == 0)
                throw std::overflow_error("Native ring submission ID overflow");
            command.submission_id = next_submission++;
            command.flags =
                profiled ? command.flags | WMFS_RING_RECORD_FLAG_PROFILE
                         : command.flags & ~WMFS_RING_RECORD_FLAG_PROFILE;
            submission->expected_flags = command.flags;
            pending.emplace(command.submission_id, submission);
        }

        RingSubmissionMetrics metrics{};
        try {
            std::lock_guard producer_lock(producer_mutex);
            const auto producer_started_ns =
                profiled ? monotonic_nanoseconds() : 0;
            if (profiled)
                command.profile.command_published_ns = monotonic_nanoseconds();
            if (!commands->try_push(command)) {
                const auto wait_started_ns =
                    profiled ? monotonic_nanoseconds() : 0;
                const auto result = commands->push(command, deadline);
                if (result != RingWaitResult::success)
                    throw std::runtime_error(
                        result == RingWaitResult::timeout
                            ? "Native command ring deadline expired"
                            : "Native command ring closed");
                if (profiled)
                    metrics.backpressure_wait_ns =
                        monotonic_nanoseconds() - wait_started_ns;
            }
            if (profiled) {
                const auto published_ns = monotonic_nanoseconds();
                metrics.submission_queue_ns =
                    ordered_delta(producer_started_ns, submitted_ns);
                metrics.enqueue_ns =
                    ordered_delta(published_ns, producer_started_ns);
            }
        } catch (...) {
            {
                std::lock_guard lock(pending_mutex);
                pending.erase(command.submission_id);
            }
            fail_ring(std::current_exception());
            throw;
        }

        std::unique_lock lock(submission->mutex);
        if (!submission->ready.wait_until(
                lock, deadline, [&] { return submission->completed; })) {
            lock.unlock();
            const auto error = std::make_exception_ptr(
                std::runtime_error("Native ring completion deadline expired"));
            fail_ring(error);
            std::rethrow_exception(error);
        }
        if (submission->error)
            std::rethrow_exception(submission->error);

        RingSubmissionResult result;
        result.completion = submission->completion;
        if (command.kind == WMFS_RING_COMMAND_INVOKE) {
            std::lock_guard mapping_lock(mutex);
            for (auto item = mappings.begin(); item != mappings.end();) {
                const auto &mapping = item->second;
                if (!mapping.arena && mapping.writable &&
                    mapping.invocation_id == command.invocation_id)
                    item = mappings.erase(item);
                else
                    ++item;
            }
        }
        if (profiled) {
            const auto returned_ns = monotonic_nanoseconds();
            const auto &worker = result.completion.profile;
            metrics.round_trip_ns = ordered_delta(returned_ns, submitted_ns);
            metrics.command_wakeup_ns = ordered_delta(
                worker.worker_dequeued_ns, worker.command_published_ns);
            metrics.worker_queue_ns = ordered_delta(worker.worker_started_ns,
                                                    worker.worker_dequeued_ns);
            metrics.worker_kernel_ns = worker.worker_kernel_ns;
            metrics.completion_wakeup_ns =
                ordered_delta(submission->completion_consumed_ns,
                              worker.completion_published_ns);
            metrics.result_materialization_ns =
                ordered_delta(returned_ns, submission->completion_consumed_ns);
            result.metrics = metrics;
        }
        return result;
    }

    std::size_t transfer_batched(const std::vector<Mapping> &values,
                                 const std::vector<bool> &maps,
                                 std::vector<int> fds) {
        std::size_t fd_index = 0;
        std::size_t batches = 0;
        try {
            for (std::size_t offset = 0; offset < values.size();
                 offset += WMFS_CONTROL_MAX_FD_ENTRIES) {
                const auto end = std::min<std::size_t>(
                    values.size(), offset + WMFS_CONTROL_MAX_FD_ENTRIES);
                std::vector<Mapping> batch_values(values.begin() + offset,
                                                  values.begin() + end);
                std::vector<bool> batch_maps(maps.begin() + offset,
                                             maps.begin() + end);
                const auto count = static_cast<std::size_t>(
                    std::count(batch_maps.begin(), batch_maps.end(), true));
                std::vector<int> batch_fds(fds.begin() + fd_index,
                                           fds.begin() + fd_index + count);
                fd_index += count;
                transfer(batch_values, batch_maps, std::move(batch_fds));
                ++batches;
            }
        } catch (...) {
            for (; fd_index < fds.size(); ++fd_index)
                ::close(fds[fd_index]);
            throw;
        }
        return batches;
    }

    void lifecycle_round_trip(std::uint16_t request_kind,
                              std::uint16_t response_kind) {
        std::array<std::uint8_t, WMFS_CONTROL_FRAME_HEADER_SIZE> packet{};
        const auto request_id = next_request++;
        wmfs_control_mutable_bytes_v1 output{packet.data(), packet.size(), 0};
        if (wmfs_control_encode_empty_v1(request_kind, request_id, &output) !=
            0)
            throw std::runtime_error("Cannot encode native lifecycle request");
        ssize_t sent;
        do {
            sent = ::send(lifecycle.get(), packet.data(), output.size,
                          MSG_NOSIGNAL);
        } while (sent < 0 && errno == EINTR);
        if (sent != static_cast<ssize_t>(output.size))
            throw std::runtime_error("Cannot send native lifecycle request");
        ssize_t received;
        do {
            received = ::recv(lifecycle.get(), packet.data(), packet.size(), 0);
        } while (received < 0 && errno == EINTR);
        std::uint64_t response_id = 0;
        const wmfs_control_bytes_v1 response{
            packet.data(),
            received > 0 ? static_cast<std::size_t>(received) : 0};
        const auto decoded =
            received > 0 ? wmfs_control_decode_empty_v1(response, response_kind,
                                                        &response_id)
                         : -1;
        if (received <= 0 || decoded != 0 || response_id != request_id)
            throw std::runtime_error(
                "Invalid native lifecycle response (bytes=" +
                std::to_string(received) +
                ", decode=" + std::to_string(decoded) +
                ", request=" + std::to_string(request_id) +
                ", response=" + std::to_string(response_id) + ", errno=" +
                std::to_string(errno) + ":" + std::strerror(errno) + ")");
    }

    void close() {
        std::lock_guard lock(mutex);
        if (closed)
            return;
        closed = true;
        std::exception_ptr error;
        try {
            lifecycle_round_trip(WMFS_CONTROL_SHUTDOWN,
                                 WMFS_CONTROL_SHUTDOWN_ACK);
        } catch (...) {
            error = std::current_exception();
        }
        ::shutdown(lifecycle.get(), SHUT_RDWR);
        ::shutdown(control.get(), SHUT_RDWR);
        if (commands)
            commands->close();
        if (completions)
            completions->close();
        if (completion_thread.joinable())
            completion_thread.join();
        mappings.clear();
        if (error)
            std::rethrow_exception(error);
    }

    UniqueFd lifecycle;
    UniqueFd control;
    std::unique_ptr<RingProducer> commands;
    std::unique_ptr<RingConsumer> completions;
    std::thread completion_thread;
    std::uint64_t generation;
    std::uint64_t next_transfer{1};
    std::uint64_t next_request{1};
    std::unordered_map<MappingKey, Mapping, MappingKeyHash> mappings;
    std::uint64_t transfers{};
    std::uint64_t mapping_batches{};
    std::uint64_t retirements{};
    std::uint64_t retirement_batches{};
    bool closed{};
    mutable std::mutex mutex;
    std::mutex producer_mutex;
    std::mutex pending_mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<PendingSubmission>>
        pending;
    std::uint64_t next_submission{1};
    std::exception_ptr ring_error;
};

Session::Session(int lifecycle_fd, int control_fd,
                 std::uint64_t expected_fingerprint,
                 double startup_timeout_seconds, double request_timeout_seconds,
                 double fd_transfer_timeout_seconds,
                 std::uint64_t ring_generation, std::uint32_t ring_capacity,
                 int command_ring_fd, int command_data_fd, int command_space_fd,
                 int completion_ring_fd, int completion_data_fd,
                 int completion_space_fd)
    : impl_(std::make_unique<Impl>(
          lifecycle_fd, control_fd, expected_fingerprint, ring_generation,
          ring_capacity, startup_timeout_seconds, request_timeout_seconds,
          fd_transfer_timeout_seconds, command_ring_fd, command_data_fd,
          command_space_fd, completion_ring_fd, completion_data_fd,
          completion_space_fd)) {}

Session::~Session() = default;

bool Session::mapping_required(const Mapping &mapping) const {
    std::lock_guard lock(impl_->mutex);
    const auto item =
        impl_->mappings.find(MappingKey{mapping.buffer_id, mapping.generation});
    return item == impl_->mappings.end() ||
           (!item->second.writable && mapping.writable);
}

void Session::map_buffer(const Mapping &mapping, int fd) {
    map_buffers({{mapping, fd}});
}

std::vector<bool>
Session::map_buffers(std::vector<std::pair<Mapping, int>> mappings) {
    std::lock_guard lock(impl_->mutex);
    std::vector<Mapping> values;
    std::vector<bool> maps;
    std::vector<int> fds;
    std::vector<bool> result;
    for (auto &item : mappings) {
        const MappingKey key{item.first.buffer_id, item.first.generation};
        const auto existing = impl_->mappings.find(key);
        const bool needed = existing == impl_->mappings.end() ||
                            (!existing->second.writable && item.first.writable);
        result.push_back(needed);
        if (!needed) {
            ::close(item.second);
            continue;
        }
        if (existing != impl_->mappings.end()) {
            values.push_back(existing->second);
            maps.push_back(false);
        }
        values.push_back(item.first);
        maps.push_back(true);
        fds.push_back(item.second);
        impl_->mappings[key] = item.first;
    }
    if (!values.empty()) {
        const auto batches =
            impl_->transfer_batched(values, maps, std::move(fds));
        impl_->transfers += std::count(maps.begin(), maps.end(), true);
        impl_->mapping_batches += batches;
    }
    return result;
}

void Session::retire_buffer(const Mapping &mapping) {
    retire_buffers({mapping});
}

void Session::retire_buffers(const std::vector<Mapping> &mappings) {
    if (mappings.empty())
        return;
    std::lock_guard lock(impl_->mutex);
    std::vector<Mapping> active;
    for (const auto &mapping : mappings) {
        const auto item = impl_->mappings.find(
            MappingKey{mapping.buffer_id, mapping.generation});
        if (item != impl_->mappings.end() && !item->second.arena)
            active.push_back(item->second);
    }
    if (active.empty())
        return;
    std::vector<bool> maps(active.size(), false);
    const auto batches = impl_->transfer_batched(active, maps, {});
    for (const auto &mapping : active)
        impl_->mappings.erase(
            MappingKey{mapping.buffer_id, mapping.generation});
    impl_->retirements += active.size();
    impl_->retirement_batches += batches;
}

void Session::abort_invocation(std::uint64_t invocation_id) {
    std::vector<Mapping> expired;
    {
        std::lock_guard lock(impl_->mutex);
        for (const auto &item : impl_->mappings)
            if (!item.second.arena && item.second.writable &&
                item.second.invocation_id == invocation_id)
                expired.push_back(item.second);
    }
    retire_buffers(expired);
}

InvocationOutcome Session::invoke(std::uint64_t, std::uint32_t,
                                  const TensorDescriptors &,
                                  const TensorDescriptors &,
                                  const std::vector<ScalarArgument> &) {
    throw std::runtime_error("Operations are available only through rings");
}

InvocationProfile
Session::invoke_profiled(std::uint64_t, std::uint32_t,
                         const TensorDescriptors &, const TensorDescriptors &,
                         const std::vector<ScalarArgument> &) {
    throw std::runtime_error("Operations are available only through rings");
}

OutputPlanningResult
Session::plan_outputs(std::uint64_t, std::uint32_t, const TensorDescriptors &,
                      const std::vector<ScalarArgument> &) {
    throw std::runtime_error("Output planning is available only through rings");
}

RingSubmissionResult Session::submit_ring(wmfs_ring_record_v1 command,
                                          double timeout_seconds,
                                          bool profiled) {
    return impl_->submit_ring(command, timeout_seconds, profiled);
}

void Session::ping(std::uint64_t) {
    std::lock_guard lock(impl_->mutex);
    impl_->lifecycle_round_trip(WMFS_CONTROL_PING, WMFS_CONTROL_PONG);
}

std::vector<std::uint8_t> Session::metadata() { return {}; }
std::vector<std::uint8_t> Session::environment() { return {}; }
void Session::close() { impl_->close(); }
std::uint64_t Session::transfer_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->transfers;
}
std::uint64_t Session::mapping_batch_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->mapping_batches;
}
std::uint64_t Session::retirement_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->retirements;
}
std::uint64_t Session::retirement_batch_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->retirement_batches;
}

} // namespace wmfs::native
