#include "wmfs/ring.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <limits>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>

namespace wmfs {
namespace {

static_assert(offsetof(wmfs_ring_header_v1, producer) %
                  alignof(std::uint64_t) ==
              0);
static_assert(offsetof(wmfs_ring_header_v1, consumer) %
                  alignof(std::uint64_t) ==
              0);
static_assert(__atomic_always_lock_free(sizeof(std::uint64_t), nullptr),
              "ring counters require lock-free 64-bit atomics");

[[noreturn]] void fail_errno(const char *operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

[[noreturn]] void corrupt(const char *reason) {
    throw std::runtime_error(std::string("invalid ring: ") + reason);
}

std::size_t mapping_size(std::uint32_t capacity) {
    if (capacity == 0)
        throw std::invalid_argument("ring capacity must be nonzero");
    constexpr auto maximum =
        (std::numeric_limits<std::size_t>::max() - WMFS_RING_HEADER_SIZE) /
        WMFS_RING_RECORD_SIZE;
    if (capacity > maximum)
        throw std::overflow_error("ring mapping size overflow");
    return WMFS_RING_HEADER_SIZE +
           static_cast<std::size_t>(capacity) * WMFS_RING_RECORD_SIZE;
}

class Mapping {
  public:
    Mapping() = default;
    Mapping(int fd, std::size_t size) : size_(size) {
        address_ =
            ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            fail_errno("mmap ring");
        }
    }
    ~Mapping() {
        if (address_ != nullptr)
            ::munmap(address_, size_);
    }
    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
    Mapping(Mapping &&other) noexcept
        : address_(std::exchange(other.address_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}
    Mapping &operator=(Mapping &&other) noexcept {
        if (this != &other) {
            if (address_ != nullptr)
                ::munmap(address_, size_);
            address_ = std::exchange(other.address_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    [[nodiscard]] void *get() const noexcept { return address_; }

  private:
    void *address_ = nullptr;
    std::size_t size_ = 0;
};

std::uint64_t acquire(const std::uint64_t *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void release(std::uint64_t *value, std::uint64_t next) {
    __atomic_store_n(value, next, __ATOMIC_RELEASE);
}

std::uint32_t acquire_flags(const std::uint32_t *value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void close_header(wmfs_ring_header_v1 *header) {
    __atomic_fetch_or(&header->flags, WMFS_RING_HEADER_FLAG_CLOSED,
                      __ATOMIC_RELEASE);
}

void notify(int fd) {
    const std::uint64_t one = 1;
    ssize_t result;
    do {
        result = ::write(fd, &one, sizeof(one));
    } while (result < 0 && errno == EINTR);
    if (result < 0 && errno != EAGAIN)
        fail_errno("write ring eventfd");
}

void drain(int fd) {
    std::uint64_t value;
    ssize_t result;
    do {
        result = ::read(fd, &value, sizeof(value));
    } while (result < 0 && errno == EINTR);
    if (result < 0 && errno != EAGAIN)
        fail_errno("read ring eventfd");
}

bool all_zero(const void *data, std::size_t size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    return std::all_of(bytes, bytes + size,
                       [](unsigned char byte) { return byte == 0; });
}

void validate_header(const wmfs_ring_header_v1 &header,
                     std::uint64_t expected_generation,
                     std::uint32_t expected_capacity) {
    if (header.magic != WMFS_RING_MAGIC)
        corrupt("header magic");
    if (header.abi_major != WMFS_RING_ABI_MAJOR ||
        header.abi_minor != WMFS_RING_ABI_MINOR)
        corrupt("header ABI version");
    if (header.header_size != WMFS_RING_HEADER_SIZE ||
        header.record_size != WMFS_RING_RECORD_SIZE)
        corrupt("header geometry");
    if (header.capacity != expected_capacity || header.capacity == 0)
        corrupt("header capacity");
    if (header.session_generation != expected_generation)
        corrupt("session generation");
    const auto flags = acquire_flags(&header.flags);
    constexpr std::uint32_t allowed =
        WMFS_RING_HEADER_FLAG_INITIALIZED | WMFS_RING_HEADER_FLAG_CLOSED;
    if ((flags & WMFS_RING_HEADER_FLAG_INITIALIZED) == 0 ||
        (flags & ~allowed) != 0)
        corrupt("header flags");
    if (!all_zero(header.reserved0, sizeof(header.reserved0)) ||
        !all_zero(header.reserved1, sizeof(header.reserved1)) ||
        !all_zero(header.reserved2, sizeof(header.reserved2)))
        corrupt("reserved header bytes");
}

void validate_counters(const wmfs_ring_header_v1 &header,
                       std::uint64_t producer, std::uint64_t consumer) {
    if (producer < consumer || producer - consumer > header.capacity)
        corrupt("impossible counters");
}

void validate_file(int fd, std::size_t expected_size) {
    struct stat status{};
    if (::fstat(fd, &status) < 0)
        fail_errno("fstat ring");
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != expected_size)
        corrupt("file size");
}

struct EndpointStorage {
    UniqueFd ring_fd;
    UniqueFd data_fd;
    UniqueFd space_fd;
    Mapping mapping;
    wmfs_ring_header_v1 *header;
    wmfs_ring_record_v1 *records;
    std::uint64_t generation;
    std::uint32_t capacity;

    EndpointStorage(UniqueFd ring, UniqueFd data, UniqueFd space,
                    std::uint64_t expected_generation)
        : ring_fd(std::move(ring)), data_fd(std::move(data)),
          space_fd(std::move(space)),
          mapping(map_checked(ring_fd.get(), expected_generation)),
          header(static_cast<wmfs_ring_header_v1 *>(mapping.get())),
          records(reinterpret_cast<wmfs_ring_record_v1 *>(
              static_cast<unsigned char *>(mapping.get()) +
              WMFS_RING_HEADER_SIZE)),
          generation(expected_generation), capacity(header->capacity) {
        if (!ring_fd || !data_fd || !space_fd)
            throw std::invalid_argument("ring endpoint has an invalid fd");
        validate_header(*header, generation, capacity);
    }

    static Mapping map_checked(int fd, std::uint64_t generation) {
        struct stat status{};
        if (fd < 0)
            throw std::invalid_argument("ring endpoint has an invalid fd");
        if (::fstat(fd, &status) < 0)
            fail_errno("fstat ring");
        if (status.st_size < static_cast<off_t>(WMFS_RING_HEADER_SIZE))
            corrupt("file size");
        Mapping map(fd, static_cast<std::size_t>(status.st_size));
        const auto *header = static_cast<wmfs_ring_header_v1 *>(map.get());
        const auto size = mapping_size(header->capacity);
        if (size != static_cast<std::size_t>(status.st_size))
            corrupt("file size");
        validate_header(*header, generation, header->capacity);
        return map;
    }

    void validate() const { validate_header(*header, generation, capacity); }
    [[nodiscard]] bool closed() const {
        return (acquire_flags(&header->flags) & WMFS_RING_HEADER_FLAG_CLOSED) !=
               0;
    }
    void close() {
        validate();
        close_header(header);
        notify(data_fd.get());
        notify(space_fd.get());
    }
};

int poll_timeout(RingDeadline deadline) {
    if (deadline == RingDeadline::max())
        return -1;
    const auto now = std::chrono::steady_clock::now();
    if (deadline <= now)
        return 0;
    const auto remaining = deadline - now;
    const auto millis =
        std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    auto count = millis.count() + (millis < remaining ? 1 : 0);
    return static_cast<int>(std::min<std::int64_t>(count, INT_MAX));
}

bool wait_event(int fd, RingDeadline deadline) {
    pollfd descriptor{fd, POLLIN, 0};
    for (;;) {
        const int result = ::poll(&descriptor, 1, poll_timeout(deadline));
        if (result > 0)
            return true;
        if (result == 0)
            return false;
        if (errno != EINTR)
            fail_errno("poll ring eventfd");
    }
}

void validate_record_generation(const wmfs_ring_record_v1 &record,
                                std::uint64_t generation) {
    if (record.session_generation != generation)
        corrupt("record session generation");
    if (record.tensor_count > WMFS_RING_MAX_TENSORS ||
        record.scalar_count > WMFS_RING_MAX_SCALARS ||
        record.planned_output_count > WMFS_RING_MAX_PLANNED_OUTPUTS)
        corrupt("record descriptor counts");
    if (record.reserved0 != 0 || record.reserved1 != 0 ||
        !all_zero(record.reserved_header, sizeof(record.reserved_header)) ||
        !all_zero(record.reserved_tail, sizeof(record.reserved_tail)))
        corrupt("reserved record bytes");
    for (const auto &tensor : record.tensors) {
        if (tensor.reserved0 != 0)
            corrupt("reserved tensor descriptor bytes");
    }
    for (const auto &scalar : record.scalars) {
        if (scalar.reserved0 != 0)
            corrupt("reserved scalar descriptor bytes");
    }
    for (const auto &output : record.planned_outputs) {
        if (output.reserved0 != 0)
            corrupt("reserved output descriptor bytes");
    }
    if (record.error.reserved0 != 0)
        corrupt("reserved error bytes");
}

} // namespace

struct RingOwner::Impl {
    UniqueFd ring_fd;
    UniqueFd data_fd;
    UniqueFd space_fd;
    Mapping mapping;
    wmfs_ring_header_v1 *header;
    std::uint64_t generation;
    std::uint32_t capacity;
};

struct RingProducer::Impl : EndpointStorage {
    using EndpointStorage::EndpointStorage;
};

struct RingConsumer::Impl : EndpointStorage {
    using EndpointStorage::EndpointStorage;
};

RingOwner::RingOwner(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
RingOwner::~RingOwner() = default;
RingOwner::RingOwner(RingOwner &&) noexcept = default;
RingOwner &RingOwner::operator=(RingOwner &&) noexcept = default;

RingOwner RingOwner::create(std::uint32_t capacity,
                            std::uint64_t session_generation) {
    const auto size = mapping_size(capacity);
    const int raw_ring =
        static_cast<int>(::syscall(SYS_memfd_create, "wmfs-ring", MFD_CLOEXEC));
    if (raw_ring < 0)
        fail_errno("memfd_create ring");
    UniqueFd ring(raw_ring);
    if (::ftruncate(ring.get(), static_cast<off_t>(size)) < 0)
        fail_errno("ftruncate ring");
    UniqueFd data(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC));
    if (!data)
        fail_errno("eventfd ring data");
    UniqueFd space(::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC));
    if (!space)
        fail_errno("eventfd ring space");

    Mapping mapping(ring.get(), size);
    std::memset(mapping.get(), 0, size);
    auto *header = static_cast<wmfs_ring_header_v1 *>(mapping.get());
    header->magic = WMFS_RING_MAGIC;
    header->abi_major = WMFS_RING_ABI_MAJOR;
    header->abi_minor = WMFS_RING_ABI_MINOR;
    header->header_size = WMFS_RING_HEADER_SIZE;
    header->record_size = WMFS_RING_RECORD_SIZE;
    header->capacity = capacity;
    header->session_generation = session_generation;
    __atomic_store_n(&header->flags, WMFS_RING_HEADER_FLAG_INITIALIZED,
                     __ATOMIC_RELEASE);
    validate_file(ring.get(), size);
    validate_header(*header, session_generation, capacity);
    return RingOwner(std::make_unique<Impl>(
        Impl{std::move(ring), std::move(data), std::move(space),
             std::move(mapping), header, session_generation, capacity}));
}

int RingOwner::ring_fd() const noexcept { return impl_->ring_fd.get(); }
int RingOwner::data_event_fd() const noexcept { return impl_->data_fd.get(); }
int RingOwner::space_event_fd() const noexcept { return impl_->space_fd.get(); }
UniqueFd RingOwner::duplicate_ring_fd() const {
    return impl_->ring_fd.duplicate_cloexec();
}
UniqueFd RingOwner::duplicate_data_event_fd() const {
    return impl_->data_fd.duplicate_cloexec();
}
UniqueFd RingOwner::duplicate_space_event_fd() const {
    return impl_->space_fd.duplicate_cloexec();
}
RingProducer RingOwner::producer() const {
    return RingProducer::borrow(duplicate_ring_fd(), duplicate_data_event_fd(),
                                duplicate_space_event_fd(), impl_->generation);
}
RingConsumer RingOwner::consumer() const {
    return RingConsumer::borrow(duplicate_ring_fd(), duplicate_data_event_fd(),
                                duplicate_space_event_fd(), impl_->generation);
}
void RingOwner::close() {
    validate_header(*impl_->header, impl_->generation, impl_->capacity);
    close_header(impl_->header);
    notify(impl_->data_fd.get());
    notify(impl_->space_fd.get());
}

RingProducer::RingProducer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
RingProducer::~RingProducer() = default;
RingProducer::RingProducer(RingProducer &&) noexcept = default;
RingProducer &RingProducer::operator=(RingProducer &&) noexcept = default;
RingProducer RingProducer::borrow(UniqueFd ring_fd, UniqueFd data_event_fd,
                                  UniqueFd space_event_fd,
                                  std::uint64_t expected_generation) {
    return RingProducer(
        std::make_unique<Impl>(std::move(ring_fd), std::move(data_event_fd),
                               std::move(space_event_fd), expected_generation));
}
bool RingProducer::try_push(const wmfs_ring_record_v1 &record) {
    impl_->validate();
    if (impl_->closed())
        return false;
    validate_record_generation(record, impl_->generation);
    const auto producer = acquire(&impl_->header->producer);
    const auto consumer = acquire(&impl_->header->consumer);
    validate_counters(*impl_->header, producer, consumer);
    if (producer - consumer == impl_->capacity)
        return false;
    if (producer == std::numeric_limits<std::uint64_t>::max())
        corrupt("producer counter overflow");
    auto *slot = &impl_->records[producer % impl_->capacity];
    std::memset(slot, 0, sizeof(*slot));
    std::memcpy(slot, &record, sizeof(record));
    release(&impl_->header->producer, producer + 1);
    notify(impl_->data_fd.get());
    return true;
}
RingWaitResult RingProducer::push(const wmfs_ring_record_v1 &record,
                                  RingDeadline deadline) {
    for (;;) {
        if (try_push(record))
            return RingWaitResult::success;
        if (impl_->closed())
            return RingWaitResult::closed;
        drain(impl_->space_fd.get());
        if (try_push(record))
            return RingWaitResult::success;
        if (impl_->closed())
            return RingWaitResult::closed;
        if (!wait_event(impl_->space_fd.get(), deadline))
            return RingWaitResult::timeout;
    }
}
void RingProducer::close() { impl_->close(); }
int RingProducer::ring_fd() const noexcept { return impl_->ring_fd.get(); }
int RingProducer::data_event_fd() const noexcept {
    return impl_->data_fd.get();
}
int RingProducer::space_event_fd() const noexcept {
    return impl_->space_fd.get();
}
UniqueFd RingProducer::duplicate_ring_fd() const {
    return impl_->ring_fd.duplicate_cloexec();
}
UniqueFd RingProducer::duplicate_data_event_fd() const {
    return impl_->data_fd.duplicate_cloexec();
}
UniqueFd RingProducer::duplicate_space_event_fd() const {
    return impl_->space_fd.duplicate_cloexec();
}

RingConsumer::RingConsumer(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}
RingConsumer::~RingConsumer() = default;
RingConsumer::RingConsumer(RingConsumer &&) noexcept = default;
RingConsumer &RingConsumer::operator=(RingConsumer &&) noexcept = default;
RingConsumer RingConsumer::borrow(UniqueFd ring_fd, UniqueFd data_event_fd,
                                  UniqueFd space_event_fd,
                                  std::uint64_t expected_generation) {
    return RingConsumer(
        std::make_unique<Impl>(std::move(ring_fd), std::move(data_event_fd),
                               std::move(space_event_fd), expected_generation));
}
bool RingConsumer::try_pop(wmfs_ring_record_v1 &record) {
    impl_->validate();
    const auto producer = acquire(&impl_->header->producer);
    const auto consumer = acquire(&impl_->header->consumer);
    validate_counters(*impl_->header, producer, consumer);
    if (producer == consumer)
        return false;
    if (consumer == std::numeric_limits<std::uint64_t>::max())
        corrupt("consumer counter overflow");
    const auto *slot = &impl_->records[consumer % impl_->capacity];
    std::memcpy(&record, slot, sizeof(record));
    validate_record_generation(record, impl_->generation);
    release(&impl_->header->consumer, consumer + 1);
    notify(impl_->space_fd.get());
    return true;
}
RingWaitResult RingConsumer::pop(wmfs_ring_record_v1 &record,
                                 RingDeadline deadline) {
    for (;;) {
        if (try_pop(record))
            return RingWaitResult::success;
        if (impl_->closed())
            return RingWaitResult::closed;
        drain(impl_->data_fd.get());
        if (try_pop(record))
            return RingWaitResult::success;
        if (impl_->closed())
            return RingWaitResult::closed;
        if (!wait_event(impl_->data_fd.get(), deadline))
            return RingWaitResult::timeout;
    }
}
void RingConsumer::close() { impl_->close(); }
int RingConsumer::ring_fd() const noexcept { return impl_->ring_fd.get(); }
int RingConsumer::data_event_fd() const noexcept {
    return impl_->data_fd.get();
}
int RingConsumer::space_event_fd() const noexcept {
    return impl_->space_fd.get();
}
UniqueFd RingConsumer::duplicate_ring_fd() const {
    return impl_->ring_fd.duplicate_cloexec();
}
UniqueFd RingConsumer::duplicate_data_event_fd() const {
    return impl_->data_fd.duplicate_cloexec();
}
UniqueFd RingConsumer::duplicate_space_event_fd() const {
    return impl_->space_fd.duplicate_cloexec();
}

} // namespace wmfs
