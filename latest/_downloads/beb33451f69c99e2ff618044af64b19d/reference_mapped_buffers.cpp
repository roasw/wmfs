#include "wmfs/protocol/control.h"
#include "wmfs/protocol/ring.h"
#include "wmfs/reference/mapped_buffers.hpp"
#include "wmfs/unique_fd.hpp"

#include <ATen/ops/from_blob.h>

#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <limits>
#include <list>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace wmfs::reference {
namespace {

constexpr std::size_t MAX_CONTROL_MESSAGE_BYTES = 64 * 1024;
constexpr std::size_t MAX_CACHED_VIEWS = 64;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

[[noreturn]] void fail_errno(const char *operation) {
    throw std::runtime_error(std::string(operation) + ": " +
                             std::strerror(errno));
}

struct ViewKey {
    std::uint64_t allocation_id;
    std::uint64_t offset;
    std::uint64_t byte_length;
    std::uint32_t dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;

    bool operator==(const ViewKey &) const = default;
};

struct MappedRegion {
    MappedRegion(void *mapped_address, std::size_t mapped_length)
        : address(mapped_address), length(mapped_length) {}

    ~MappedRegion() { ::munmap(address, length); }

    void *address;
    std::size_t length;
};

struct Mapping {
    Mapping(MappingSpec specification,
            std::shared_ptr<MappedRegion> mapped_region)
        : spec(specification), region(std::move(mapped_region)) {}

    MappingSpec spec;
    std::shared_ptr<MappedRegion> region;
    std::list<std::pair<ViewKey, at::Tensor>> views;
};

std::pair<at::ScalarType, std::uint64_t> dtype_info(std::uint32_t dtype) {
    switch (dtype) {
    case WMFS_RING_DTYPE_FLOAT32:
        return {at::kFloat, 4};
    case WMFS_RING_DTYPE_FLOAT64:
        return {at::kDouble, 8};
    case WMFS_RING_DTYPE_INT64:
        return {at::kLong, 8};
    case WMFS_RING_DTYPE_UINT8:
        return {at::kByte, 1};
    }
    fail("Unsupported tensor dtype");
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        fail("Tensor descriptor arithmetic overflow");
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        fail("Tensor descriptor arithmetic overflow");
    }
    return left * right;
}

void send_acknowledgement(int socket_fd, std::uint64_t request_id,
                          std::uint64_t transfer_id,
                          std::uint64_t session_generation, const char *error) {
    std::array<std::uint8_t, WMFS_CONTROL_FRAME_HEADER_SIZE +
                                 WMFS_CONTROL_FD_ACK_FIXED_SIZE +
                                 WMFS_CONTROL_MAX_ERROR_BYTES>
        payload{};
    const auto error_size =
        error == nullptr ? 0
                         : std::min<std::size_t>(std::strlen(error),
                                                 WMFS_CONTROL_MAX_ERROR_BYTES);
    wmfs_control_fd_ack_v1 acknowledgement{};
    acknowledgement.transfer_id = transfer_id;
    acknowledgement.session_generation = session_generation;
    acknowledgement.status = error == nullptr
                                 ? WMFS_CONTROL_STATUS_OK
                                 : WMFS_CONTROL_STATUS_INVALID_ARGUMENT;
    acknowledgement.error_length = error_size;
    wmfs_control_mutable_bytes_v1 output{payload.data(), payload.size(), 0};
    wmfs_control_bytes_v1 error_bytes{
        reinterpret_cast<const std::uint8_t *>(error), error_size};
    if (wmfs_control_encode_fd_ack_v1(request_id, &acknowledgement, error_bytes,
                                      &output) != 0)
        fail("Cannot encode buffer acknowledgement");
    auto sent = ::send(socket_fd, payload.data(), output.size, MSG_NOSIGNAL);
    if (sent < 0) {
        fail_errno("send buffer acknowledgement");
    }
    if (static_cast<std::size_t>(sent) != output.size) {
        fail("Buffer acknowledgement was truncated");
    }
}

std::vector<UniqueFd> extract_descriptors(msghdr &message) {
    std::vector<UniqueFd> descriptors;
    for (auto *item = CMSG_FIRSTHDR(&message); item != nullptr;
         item = CMSG_NXTHDR(&message, item)) {
        if (item->cmsg_level != SOL_SOCKET || item->cmsg_type != SCM_RIGHTS)
            fail("FD transfer contains unsupported ancillary data");
        if (item->cmsg_len < CMSG_LEN(0))
            fail("FD transfer contains malformed ancillary data");
        auto data_length = item->cmsg_len - CMSG_LEN(0);
        if (data_length % sizeof(int) != 0)
            fail("FD transfer contains a malformed descriptor array");
        auto count = data_length / sizeof(int);
        auto *values = reinterpret_cast<int *>(CMSG_DATA(item));
        for (std::size_t index = 0; index < count; ++index) {
            descriptors.emplace_back(values[index]);
        }
    }
    return descriptors;
}

} // namespace

struct MappedBufferCache::Impl {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, std::shared_ptr<Mapping>> buffers;
};

TensorLease::TensorLease(at::Tensor tensor) : tensor_(std::move(tensor)) {}

const at::Tensor &TensorLease::tensor() const noexcept { return tensor_; }

at::Tensor &TensorLease::tensor() noexcept { return tensor_; }

MappedBufferCache::MappedBufferCache() : impl_(std::make_unique<Impl>()) {}

MappedBufferCache::~MappedBufferCache() { close(); }

void MappedBufferCache::map(MappingSpec spec, int raw_fd) {
    UniqueFd fd(raw_fd);
    if (spec.byte_length == 0 ||
        spec.byte_length > std::numeric_limits<std::size_t>::max()) {
        fail("Transferred buffer has an invalid byte length");
    }

    struct stat status{};
    if (::fstat(fd.get(), &status) < 0) {
        fail_errno("fstat transferred buffer");
    }
    if (status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != spec.byte_length) {
        fail("Transferred FD size does not match its descriptor");
    }

    auto protection = PROT_READ | (spec.writable ? PROT_WRITE : 0);
    auto *address = ::mmap(nullptr, static_cast<std::size_t>(spec.byte_length),
                           protection, MAP_SHARED, fd.get(), 0);
    if (address == MAP_FAILED) {
        fail_errno("mmap transferred buffer");
    }
    auto region = std::make_shared<MappedRegion>(
        address, static_cast<std::size_t>(spec.byte_length));
    auto candidate = std::make_shared<Mapping>(spec, std::move(region));

    std::lock_guard lock(impl_->mutex);
    auto existing = impl_->buffers.find(spec.buffer_id);
    if (existing != impl_->buffers.end()) {
        const auto &current = existing->second->spec;
        if (current.generation == spec.generation &&
            (current.writable || !spec.writable) &&
            current.arena == spec.arena &&
            (spec.arena || current.allocation_id == spec.allocation_id)) {
            return;
        }
        fail("Existing buffer mapping must be retired before remap");
    }
    impl_->buffers.emplace(spec.buffer_id, std::move(candidate));
}

void MappedBufferCache::retire(std::uint64_t buffer_id,
                               std::uint32_t generation,
                               std::uint64_t allocation_id) {
    std::shared_ptr<Mapping> retired;
    {
        std::lock_guard lock(impl_->mutex);
        auto item = impl_->buffers.find(buffer_id);
        if (item == impl_->buffers.end()) {
            return;
        }
        const auto &spec = item->second->spec;
        if (spec.generation != generation ||
            spec.allocation_id != allocation_id) {
            fail("Cannot retire a stale buffer generation");
        }
        if (spec.arena) {
            fail("Cannot retire the shared arena mapping");
        }
        retired = std::move(item->second);
        impl_->buffers.erase(item);
    }
    retired->views.clear();
}

TensorLease MappedBufferCache::tensor(const TensorDescriptor &descriptor,
                                      std::uint64_t invocation_id,
                                      bool require_writable) {
    std::lock_guard lock(impl_->mutex);
    auto item = impl_->buffers.find(descriptor.buffer_id);
    if (item == impl_->buffers.end()) {
        fail("Tensor references an unmapped buffer");
    }
    auto mapping = item->second;
    const auto &spec = mapping->spec;
    if (spec.generation != descriptor.generation) {
        fail("Tensor references a stale buffer generation");
    }
    if (!spec.arena && spec.allocation_id != descriptor.allocation_id) {
        fail("Tensor references a stale logical allocation");
    }
    if (require_writable && !spec.writable) {
        fail("Tensor output is not mapped writable");
    }
    if (require_writable && !spec.arena &&
        spec.invocation_id != invocation_id) {
        fail("Tensor output is outside this invocation");
    }

    auto [scalar_type, item_size] = dtype_info(descriptor.dtype);
    const auto &shape_reader = descriptor.shape;
    const auto &stride_reader = descriptor.strides;
    if (shape_reader.size() == 0 ||
        shape_reader.size() != stride_reader.size() ||
        shape_reader.size() > 16) {
        fail("Tensor shape and strides have invalid ranks");
    }

    ViewKey key{
        descriptor.allocation_id,
        descriptor.offset,
        descriptor.byte_length,
        descriptor.dtype,
        {},
        {},
    };
    key.shape.reserve(shape_reader.size());
    key.strides.reserve(stride_reader.size());
    for (std::size_t index = 0; index < shape_reader.size(); ++index) {
        auto dimension = shape_reader[index];
        auto stride = stride_reader[index];
        if (dimension == 0 ||
            dimension > std::numeric_limits<std::int64_t>::max()) {
            fail("Tensor shape must be non-empty and positive");
        }
        if (stride < 0 || static_cast<std::uint64_t>(stride) % item_size != 0) {
            fail("Tensor strides must be non-negative and dtype-aligned");
        }
        key.shape.push_back(static_cast<std::int64_t>(dimension));
        key.strides.push_back(stride);
    }

    auto cached =
        std::find_if(mapping->views.begin(), mapping->views.end(),
                     [&](const auto &entry) { return entry.first == key; });
    if (cached != mapping->views.end()) {
        auto tensor = cached->second;
        mapping->views.splice(mapping->views.end(), mapping->views, cached);
        return TensorLease(std::move(tensor));
    }

    auto offset = descriptor.offset;
    auto byte_length = descriptor.byte_length;
    if (offset % item_size != 0) {
        fail("Tensor offset is not dtype-aligned");
    }
    if (byte_length == 0 ||
        checked_add(offset, byte_length) > spec.byte_length) {
        fail("Tensor byte range exceeds its mapped buffer");
    }
    auto required = item_size;
    std::vector<std::int64_t> element_strides;
    element_strides.reserve(key.strides.size());
    for (std::size_t index = 0; index < key.shape.size(); ++index) {
        required = checked_add(
            required,
            checked_multiply(static_cast<std::uint64_t>(key.shape[index] - 1),
                             static_cast<std::uint64_t>(key.strides[index])));
        element_strides.push_back(key.strides[index] /
                                  static_cast<std::int64_t>(item_size));
    }
    if (required > byte_length) {
        fail("Tensor strides exceed its declared byte range");
    }

    auto region = mapping->region;
    auto *data = static_cast<std::byte *>(region->address) + offset;
    auto tensor = at::from_blob(
        data, key.shape, element_strides,
        [region = std::move(region)](void *) mutable { region.reset(); },
        at::TensorOptions().device(at::kCPU).dtype(scalar_type));
    mapping->views.emplace_back(std::move(key), tensor);
    if (mapping->views.size() > MAX_CACHED_VIEWS) {
        mapping->views.pop_front();
    }
    return TensorLease(std::move(tensor));
}

void MappedBufferCache::finish_invocation(std::uint64_t invocation_id) {
    std::vector<std::shared_ptr<Mapping>> expired;
    {
        std::lock_guard lock(impl_->mutex);
        for (auto item = impl_->buffers.begin();
             item != impl_->buffers.end();) {
            const auto &spec = item->second->spec;
            if (spec.writable && !spec.arena &&
                spec.invocation_id == invocation_id) {
                expired.push_back(std::move(item->second));
                item = impl_->buffers.erase(item);
            } else {
                ++item;
            }
        }
    }
    for (auto &mapping : expired) {
        mapping->views.clear();
    }
}

void MappedBufferCache::close() {
    std::unordered_map<std::uint64_t, std::shared_ptr<Mapping>> buffers;
    {
        std::lock_guard lock(impl_->mutex);
        buffers.swap(impl_->buffers);
    }
    for (auto &entry : buffers) {
        entry.second->views.clear();
    }
}

void receive_buffer_transfers(int control_fd, std::uint64_t session_generation,
                              MappedBufferCache &cache) {
    std::array<std::uint8_t, WMFS_CONTROL_MAX_PACKET_BYTES> payload{};
    std::array<std::byte, CMSG_SPACE(sizeof(int) * WMFS_CONTROL_MAX_FD_ENTRIES)>
        ancillary{};
    std::array<wmfs_control_fd_entry_v1, WMFS_CONTROL_MAX_FD_ENTRIES> entries{};

    while (true) {
        iovec vector{payload.data(), payload.size()};
        msghdr message{};
        message.msg_iov = &vector;
        message.msg_iovlen = 1;
        message.msg_control = ancillary.data();
        message.msg_controllen = ancillary.size();

        ssize_t received;
        do {
            received = ::recvmsg(control_fd, &message, MSG_CMSG_CLOEXEC);
        } while (received < 0 && errno == EINTR);
        if (received == 0) {
            return;
        }
        if (received < 0) {
            if (errno == EBADF || errno == ECONNRESET) {
                return;
            }
            fail_errno("receive buffer transfer");
        }

        std::uint64_t transfer_id = 0;
        std::uint64_t request_id = 0;
        auto descriptors = extract_descriptors(message);
        try {
            if ((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
                fail("FD transfer message was truncated");
            }
            wmfs_control_fd_batch_view_v1 transfer{};
            const wmfs_control_bytes_v1 packet{
                payload.data(), static_cast<std::size_t>(received)};
            if (wmfs_control_decode_fd_batch_v1(
                    packet, &transfer, entries.data(), entries.size()) != 0)
                fail("Invalid fixed-protocol FD batch");
            request_id = transfer.request_id;
            transfer_id = transfer.batch.transfer_id;
            if (transfer.batch.session_generation != session_generation)
                fail("FD batch has the wrong session generation");
            if (descriptors.size() != transfer.batch.fd_count)
                fail(
                    "Buffer batch descriptor count does not match map entries");
            std::size_t descriptor_index = 0;
            for (std::size_t index = 0; index < transfer.batch.entry_count;
                 ++index) {
                const auto &entry = entries[index];
                if (entry.kind == WMFS_CONTROL_FD_ENTRY_MAP) {
                    MappingSpec spec{
                        entry.buffer_id,
                        static_cast<std::uint32_t>(entry.generation),
                        entry.allocation_id,
                        entry.byte_length,
                        entry.invocation_id,
                        (entry.flags & WMFS_CONTROL_FD_FLAG_WRITABLE) != 0,
                        (entry.flags & WMFS_CONTROL_FD_FLAG_ARENA) != 0,
                    };
                    cache.map(spec, descriptors[descriptor_index++].release());
                } else {
                    cache.retire(entry.buffer_id,
                                 static_cast<std::uint32_t>(entry.generation),
                                 entry.allocation_id);
                }
            }
            send_acknowledgement(control_fd, request_id, transfer_id,
                                 session_generation, nullptr);
        } catch (const std::exception &error) {
            cache.close();
            send_acknowledgement(control_fd, request_id, transfer_id,
                                 session_generation, error.what());
            throw;
        }
    }
}

} // namespace wmfs::reference
