#include "wmfs/reference/mapped_buffers.hpp"
#include "wmfs/unique_fd.hpp"

#include <capnp/message.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace {

using wmfs::UniqueFd;
using wmfs::reference::MappedBufferCache;
using wmfs::reference::MappingSpec;

constexpr std::uint64_t BUFFER_ID = 7;
constexpr std::uint32_t GENERATION = 3;
constexpr std::uint64_t ALLOCATION_ID = 11;
constexpr std::uint64_t INVOCATION_ID = 13;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

void require(bool condition, const char *message) {
    if (!condition)
        fail(message);
}

UniqueFd make_buffer(std::size_t length) {
    UniqueFd fd(::memfd_create("wmfs-mapped-region-test", MFD_CLOEXEC));
    if (!fd)
        fail(std::string("memfd_create: ") + std::strerror(errno));
    if (::ftruncate(fd.get(), static_cast<off_t>(length)) < 0)
        fail(std::string("ftruncate: ") + std::strerror(errno));

    const float values[]{1.0F, 2.0F, 3.0F, 4.0F};
    if (::pwrite(fd.get(), values, sizeof(values), 0) != sizeof(values))
        fail(std::string("pwrite: ") + std::strerror(errno));
    return fd;
}

TensorDescriptor::Reader make_descriptor(capnp::MallocMessageBuilder &message) {
    auto descriptor = message.initRoot<TensorDescriptor>();
    descriptor.setBufferId(BUFFER_ID);
    descriptor.setGeneration(GENERATION);
    descriptor.setAllocationId(ALLOCATION_ID);
    descriptor.setOffset(0);
    descriptor.setByteLength(4 * sizeof(float));
    descriptor.setDtype(DType::FLOAT32);
    auto shape = descriptor.initShape(1);
    shape.set(0, 4);
    auto strides = descriptor.initStrides(1);
    strides.set(0, sizeof(float));
    return descriptor.asReader();
}

bool is_mapped(void *address) {
    unsigned char resident = 0;
    if (::mincore(address, static_cast<std::size_t>(::sysconf(_SC_PAGESIZE)),
                  &resident) == 0) {
        return true;
    }
    if (errno == ENOMEM)
        return false;
    fail(std::string("mincore: ") + std::strerror(errno));
}

void run_case(const std::string &operation) {
    const auto page_size = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
    auto fd = make_buffer(page_size);
    MappedBufferCache cache;
    cache.map(MappingSpec{BUFFER_ID, GENERATION, ALLOCATION_ID, page_size,
                          INVOCATION_ID, true, false},
              fd.release());

    capnp::MallocMessageBuilder message;
    auto descriptor = make_descriptor(message);
    at::Tensor retained;
    {
        auto lease = cache.tensor(descriptor, INVOCATION_ID, true);
        retained = lease.tensor();
    }
    auto *address = retained.data_ptr();

    if (operation == "retire") {
        cache.retire(BUFFER_ID, GENERATION, ALLOCATION_ID);
    } else if (operation == "finish") {
        cache.finish_invocation(INVOCATION_ID);
    } else if (operation == "close") {
        cache.close();
    } else {
        fail("unknown lifecycle operation");
    }

    require(is_mapped(address),
            "lifecycle operation unmapped retained storage");
    auto *values = retained.data_ptr<float>();
    require(values[0] == 1.0F && values[3] == 4.0F,
            "retained tensor contains invalid values");
    values[1] = 9.0F;
    require(retained.data_ptr<float>()[1] == 9.0F,
            "retained tensor alias is not writable");

    retained = at::Tensor();
    require(!is_mapped(address), "storage release did not unmap mapped region");
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2)
        fail("expected one lifecycle operation");
    run_case(argv[1]);
}
