#include "wmfs/unique_fd.hpp"

#include <cerrno>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>

namespace {

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void require_closed(int fd, const char *message) {
    errno = 0;
    require(::fcntl(fd, F_GETFD) == -1 && errno == EBADF, message);
}

std::pair<wmfs::UniqueFd, wmfs::UniqueFd> make_pipe() {
    int descriptors[2];
    if (::pipe(descriptors) < 0)
        throw std::runtime_error("pipe failed");
    return {wmfs::UniqueFd(descriptors[0]), wmfs::UniqueFd(descriptors[1])};
}

void test_closure() {
    int raw_fd;
    {
        auto [read_fd, write_fd] = make_pipe();
        raw_fd = read_fd.get();
        require(read_fd && read_fd.get() >= 0, "owned descriptor is invalid");
    }
    require_closed(raw_fd, "destructor did not close descriptor");
}

void test_moves() {
    auto [read_fd, write_fd] = make_pipe();
    const int moved_fd = read_fd.get();
    wmfs::UniqueFd moved(std::move(read_fd));
    require(!read_fd && moved.get() == moved_fd,
            "move construction did not transfer ownership");

    auto [other_read, other_write] = make_pipe();
    const int replaced_fd = other_read.get();
    other_read = std::move(moved);
    require(!moved && other_read.get() == moved_fd,
            "move assignment did not transfer ownership");
    require_closed(replaced_fd,
                   "move assignment did not close the replaced descriptor");
}

void test_release_and_reset() {
    auto [read_fd, write_fd] = make_pipe();
    const int released_fd = read_fd.release();
    require(!read_fd && ::fcntl(released_fd, F_GETFD) >= 0,
            "release did not preserve the descriptor");

    auto [replacement, replacement_write] = make_pipe();
    const int replacement_fd = replacement.release();
    read_fd.reset(replacement_fd);
    read_fd.reset();
    require_closed(replacement_fd, "reset did not close descriptor");
    ::close(released_fd);
}

void test_duplicate_cloexec() {
    auto [read_fd, write_fd] = make_pipe();
    auto duplicate = read_fd.duplicate_cloexec();
    require(duplicate && duplicate.get() != read_fd.get(),
            "duplicate did not create a distinct descriptor");
    const int flags = ::fcntl(duplicate.get(), F_GETFD);
    require(flags >= 0 && (flags & FD_CLOEXEC) != 0,
            "duplicate is missing FD_CLOEXEC");
}

} // namespace

int main() {
    test_closure();
    test_moves();
    test_release_and_reset();
    test_duplicate_cloexec();
}
