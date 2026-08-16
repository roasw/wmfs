#pragma once

#include <cerrno>
#include <fcntl.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace wmfs {

/// @brief Move-only RAII ownership for a POSIX file descriptor.
class UniqueFd {
  public:
    /// @brief Take ownership of `fd`, or construct an empty owner for `-1`.
    explicit UniqueFd(int fd = -1) noexcept : fd_(fd) {}

    ~UniqueFd() { reset(); }

    UniqueFd(const UniqueFd &) = delete;
    UniqueFd &operator=(const UniqueFd &) = delete;

    UniqueFd(UniqueFd &&other) noexcept : fd_(other.release()) {}

    UniqueFd &operator=(UniqueFd &&other) noexcept {
        if (this != &other)
            reset(other.release());
        return *this;
    }

    /// @brief Return the owned descriptor without transferring ownership.
    [[nodiscard]] int get() const noexcept { return fd_; }
    /// @brief Test whether this object owns a valid descriptor.
    [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

    /// @brief Transfer ownership to the caller and leave this object empty.
    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

    /// @brief Close the current descriptor and optionally take ownership of
    /// another.
    void reset(int fd = -1) noexcept {
        if (fd_ == fd)
            return;
        const int previous = std::exchange(fd_, fd);
        if (previous >= 0)
            ::close(previous);
    }

    /// @brief Duplicate the descriptor with close-on-exec enabled.
    /// @param minimum Lowest descriptor number accepted by `fcntl`.
    /// @return A new owning descriptor.
    /// @throws std::system_error if duplication fails.
    [[nodiscard]] UniqueFd duplicate_cloexec(int minimum = 0) const {
        int duplicate;
        do {
            duplicate = ::fcntl(fd_, F_DUPFD_CLOEXEC, minimum);
        } while (duplicate < 0 && errno == EINTR);
        if (duplicate < 0) {
            throw std::system_error(errno, std::generic_category(),
                                    "duplicate file descriptor");
        }
        return UniqueFd(duplicate);
    }

  private:
    int fd_;
};

} // namespace wmfs
