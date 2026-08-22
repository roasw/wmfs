#include "wmfs/ring.hpp"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

using namespace std::chrono_literals;

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <typename Function>
void require_invalid(Function function, const char *message) {
    try {
        function();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error(message);
}

wmfs_ring_record_v1 record(std::uint64_t generation, std::uint64_t id) {
    wmfs_ring_record_v1 value{};
    value.kind = WMFS_RING_COMMAND_PING;
    value.session_generation = generation;
    value.submission_id = id;
    return value;
}

void test_wraparound_full_and_empty() {
    auto owner = wmfs::RingOwner::create(2, 7);
    auto producer = owner.producer();
    auto consumer = owner.consumer();
    wmfs_ring_record_v1 output{};
    require(!consumer.try_pop(output), "new ring was not empty");
    require(producer.try_push(record(7, 1)), "first push failed");
    require(producer.try_push(record(7, 2)), "second push failed");
    require(!producer.try_push(record(7, 3)), "full ring accepted a record");
    require(consumer.try_pop(output) && output.submission_id == 1,
            "first pop was out of order");
    require(producer.try_push(record(7, 3)), "wraparound push failed");
    require(consumer.try_pop(output) && output.submission_id == 2,
            "second pop was out of order");
    require(consumer.try_pop(output) && output.submission_id == 3,
            "wrapped pop was out of order");
    require(!consumer.try_pop(output), "drained ring was not empty");
}

void test_generation_and_header_validation() {
    auto owner = wmfs::RingOwner::create(2, 11);
    require_invalid([&] { (void)owner.producer().try_push(record(12, 1)); },
                    "record generation mismatch was accepted");
    auto malformed = record(11, 1);
    malformed.reserved_tail[0] = 1;
    require_invalid([&] { (void)owner.producer().try_push(malformed); },
                    "reserved record corruption was accepted");
    require_invalid(
        [&] {
            auto endpoint = wmfs::RingProducer::borrow(
                owner.duplicate_ring_fd(), owner.duplicate_data_event_fd(),
                owner.duplicate_space_event_fd(), 12);
        },
        "endpoint generation mismatch was accepted");

    auto *header = static_cast<wmfs_ring_header_v1 *>(
        ::mmap(nullptr, WMFS_RING_HEADER_SIZE, PROT_READ | PROT_WRITE,
               MAP_SHARED, owner.ring_fd(), 0));
    require(header != MAP_FAILED, "test mmap failed");
    header->reserved0[0] = 1;
    require_invalid([&] { (void)owner.consumer(); },
                    "reserved header corruption was accepted");
    header->reserved0[0] = 0;
    header->producer = 4;
    header->consumer = 0;
    auto consumer = owner.consumer();
    wmfs_ring_record_v1 output{};
    require_invalid([&] { (void)consumer.try_pop(output); },
                    "impossible counters were accepted");
    ::munmap(header, WMFS_RING_HEADER_SIZE);
}

void test_cross_thread_wakeup_and_ordering() {
    constexpr std::uint64_t count = 200;
    auto owner = wmfs::RingOwner::create(4, 17);
    auto producer = owner.producer();
    auto consumer = owner.consumer();
    std::thread thread([consumer = std::move(consumer)]() mutable {
        wmfs_ring_record_v1 output{};
        for (std::uint64_t id = 0; id < count; ++id) {
            require(consumer.pop(output, wmfs::RingDeadline::max()) ==
                        wmfs::RingWaitResult::success,
                    "thread pop failed");
            require(output.submission_id == id, "thread ordering failed");
        }
    });
    for (std::uint64_t id = 0; id < count; ++id)
        require(producer.push(record(17, id)) == wmfs::RingWaitResult::success,
                "thread push failed");
    thread.join();
}

void test_cross_process_wakeup_and_ordering() {
    constexpr std::uint64_t count = 100;
    auto owner = wmfs::RingOwner::create(2, 23);
    auto producer = owner.producer();
    const pid_t child = ::fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
        try {
            auto consumer = wmfs::RingConsumer::borrow(
                owner.duplicate_ring_fd(), owner.duplicate_data_event_fd(),
                owner.duplicate_space_event_fd(), 23);
            wmfs_ring_record_v1 output{};
            for (std::uint64_t id = 0; id < count; ++id) {
                if (consumer.pop(output,
                                 std::chrono::steady_clock::now() + 5s) !=
                        wmfs::RingWaitResult::success ||
                    output.submission_id != id)
                    _exit(2);
            }
            _exit(0);
        } catch (...) {
            _exit(3);
        }
    }
    for (std::uint64_t id = 0; id < count; ++id)
        require(producer.push(record(23, id),
                              std::chrono::steady_clock::now() + 5s) ==
                    wmfs::RingWaitResult::success,
                "process push failed");
    int status = 0;
    require(::waitpid(child, &status, 0) == child && WIFEXITED(status) &&
                WEXITSTATUS(status) == 0,
            "child ring consumer failed");
}

void test_timeout_and_close() {
    auto owner = wmfs::RingOwner::create(1, 29);
    auto producer = owner.producer();
    auto consumer = owner.consumer();
    wmfs_ring_record_v1 output{};
    require(consumer.pop(output, std::chrono::steady_clock::now() + 20ms) ==
                wmfs::RingWaitResult::timeout,
            "empty pop did not time out");
    require(producer.try_push(record(29, 1)), "setup push failed");
    require(
        producer.push(record(29, 2), std::chrono::steady_clock::now() + 20ms) ==
            wmfs::RingWaitResult::timeout,
        "full push did not time out");

    std::thread closer([&owner] {
        std::this_thread::sleep_for(20ms);
        owner.close();
    });
    require(
        producer.push(record(29, 2), std::chrono::steady_clock::now() + 5s) ==
            wmfs::RingWaitResult::closed,
        "close did not interrupt blocked push");
    closer.join();
    require(consumer.try_pop(output) && output.submission_id == 1,
            "close discarded an already-published record");
    require(consumer.pop(output) == wmfs::RingWaitResult::closed,
            "closed empty ring did not report closure");

    auto empty_owner = wmfs::RingOwner::create(1, 30);
    auto empty_consumer = empty_owner.consumer();
    std::thread consumer_closer([&empty_owner] {
        std::this_thread::sleep_for(20ms);
        empty_owner.close();
    });
    require(empty_consumer.pop(output, std::chrono::steady_clock::now() + 5s) ==
                wmfs::RingWaitResult::closed,
            "close did not interrupt blocked pop");
    consumer_closer.join();
}

void test_fd_flags() {
    auto owner = wmfs::RingOwner::create(1, 31);
    for (int fd :
         {owner.ring_fd(), owner.data_event_fd(), owner.space_event_fd()})
        require((::fcntl(fd, F_GETFD) & FD_CLOEXEC) != 0,
                "ring fd is missing CLOEXEC");
    for (int fd : {owner.data_event_fd(), owner.space_event_fd()})
        require((::fcntl(fd, F_GETFL) & O_NONBLOCK) != 0,
                "eventfd is missing NONBLOCK");
}

} // namespace

int main() {
    test_wraparound_full_and_empty();
    test_generation_and_header_validation();
    test_cross_thread_wakeup_and_ordering();
    test_cross_process_wakeup_and_ordering();
    test_timeout_and_close();
    test_fd_flags();
}
