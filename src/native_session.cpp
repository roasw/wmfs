#include "wmfs/native/session.hpp"

#include <capnp/message.h>
#include <capnp/rpc-twoparty.h>
#include <capnp/serialize.h>
#include <kj/async-io.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <wmfs/runtime.capnp.h>
#include <wmfs/tensor.capnp.h>

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

class OwnedFd {
  public:
    explicit OwnedFd(int fd = -1) : fd_(fd) {}
    ~OwnedFd() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }
    OwnedFd(const OwnedFd &) = delete;
    OwnedFd &operator=(const OwnedFd &) = delete;
    OwnedFd(OwnedFd &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
    int get() const { return fd_; }
    int release() { return std::exchange(fd_, -1); }

  private:
    int fd_;
};

int duplicate_fd(int fd) {
    const int duplicate = ::dup(fd);
    if (duplicate < 0) {
        throw std::runtime_error("Failed to duplicate native session socket");
    }
    return duplicate;
}

void set_socket_timeout(int fd) {
    constexpr timeval timeout{30, 0};
    if (::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) <
            0 ||
        ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) <
            0) {
        throw std::runtime_error("Failed to configure native control timeout");
    }
}

::DType dtype_from_native(TensorDType dtype) {
    switch (dtype) {
    case TensorDType::float32:
        return ::DType::FLOAT32;
    case TensorDType::float64:
        return ::DType::FLOAT64;
    case TensorDType::int64:
        return ::DType::INT64;
    case TensorDType::uint8:
        return ::DType::UINT8;
    }
    throw std::invalid_argument("Unsupported native tensor dtype");
}

void write_descriptor(::TensorDescriptor::Builder target,
                      const TensorDescriptor &source) {
    target.setBufferId(source.buffer_id);
    target.setGeneration(source.generation);
    target.setAllocationId(source.allocation_id);
    target.setOffset(source.offset);
    target.setByteLength(source.byte_length);
    target.setDtype(dtype_from_native(source.dtype));
    auto shape = target.initShape(source.shape.size());
    for (std::size_t index = 0; index < source.shape.size(); ++index) {
        shape.set(index, source.shape[index]);
    }
    auto strides = target.initStrides(source.strides.size());
    for (std::size_t index = 0; index < source.strides.size(); ++index) {
        strides.set(index, source.strides[index]);
    }
}

std::runtime_error kj_error(const kj::Exception &error) {
    return std::runtime_error(error.getDescription().cStr());
}

} // namespace

struct Session::Impl {
    struct Worker {
        explicit Worker(OwnedFd rpc_fd, OwnedFd control_fd,
                        std::uint64_t expected_fingerprint)
            : control_fd(std::move(control_fd)), io(kj::setupAsyncIo()),
              stream(io.lowLevelProvider->wrapSocketFd(
                  kj::AutoCloseFd(rpc_fd.release()))),
              client(*stream), plugin(client.bootstrap().castAs<::Plugin>()) {
            set_socket_timeout(this->control_fd.get());
            auto version =
                io.provider->getTimer()
                    .timeoutAfter(30 * kj::SECONDS,
                                  plugin.getProtocolVersionRequest().send())
                    .wait(io.waitScope);
            if (version.getVersion() != PROTOCOL_VERSION) {
                throw std::runtime_error(
                    "Worker protocol does not match native runtime");
            }
            auto metadata =
                io.provider->getTimer()
                    .timeoutAfter(30 * kj::SECONDS,
                                  plugin.getMetadataRequest().send())
                    .wait(io.waitScope);
            if (metadata.getMetadata().getFingerprint() !=
                expected_fingerprint) {
                throw std::runtime_error(
                    "Worker metadata does not match discovered plugin");
            }
        }

        void ping(std::uint64_t nonce) {
            auto request = plugin.pingRequest();
            request.setNonce(nonce);
            auto response = io.provider->getTimer()
                                .timeoutAfter(30 * kj::SECONDS, request.send())
                                .wait(io.waitScope);
            if (response.getNonce() != nonce) {
                throw std::runtime_error(
                    "Worker returned an invalid ping response");
            }
        }

        void map_buffer(const Mapping &mapping, int fd,
                        std::uint64_t transfer_id) {
            OwnedFd owned_fd(fd);
            capnp::MallocMessageBuilder message;
            auto transfer = message.initRoot<::BufferTransfer>();
            transfer.setTransferId(transfer_id);
            transfer.setInvocationId(mapping.invocation_id);
            transfer.setBufferId(mapping.buffer_id);
            transfer.setGeneration(mapping.generation);
            transfer.setAllocationId(mapping.allocation_id);
            transfer.setByteLength(mapping.byte_length);
            transfer.setWritable(mapping.writable);
            transfer.setArena(mapping.arena);
            transfer.setMap();
            send_control(message, transfer_id, owned_fd.get());
        }

        void retire_buffer(const Mapping &mapping, std::uint64_t transfer_id) {
            capnp::MallocMessageBuilder message;
            auto transfer = message.initRoot<::BufferTransfer>();
            transfer.setTransferId(transfer_id);
            transfer.setInvocationId(0);
            transfer.setBufferId(mapping.buffer_id);
            transfer.setGeneration(mapping.generation);
            transfer.setAllocationId(mapping.allocation_id);
            transfer.setByteLength(mapping.byte_length);
            transfer.setWritable(false);
            transfer.setArena(false);
            transfer.setRetire();
            send_control(message, transfer_id, -1);
        }

        void write_invocation(::KnownInvocation::Builder invocation,
                              std::uint64_t invocation_id,
                              std::uint32_t operation_id,
                              const TensorDescriptors &inputs,
                              const TensorDescriptors &outputs,
                              const std::vector<ScalarArgument> &scalars) {
            invocation.setInvocationId(invocation_id);
            invocation.setOperationId(operation_id);
            auto input_builders = invocation.initInputs(inputs.size());
            for (std::size_t index = 0; index < inputs.size(); ++index) {
                write_descriptor(input_builders[index], *inputs[index]);
            }
            auto output_builders = invocation.initOutputs(outputs.size());
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                write_descriptor(output_builders[index], *outputs[index]);
            }
            auto scalar_builders = invocation.initScalars(scalars.size());
            for (std::size_t index = 0; index < scalars.size(); ++index) {
                const auto &source = scalars[index];
                auto target = scalar_builders[index];
                target.setParameter(source.parameter);
                switch (source.kind) {
                case ScalarKind::boolean:
                    target.setBoolean(source.boolean_value);
                    break;
                case ScalarKind::float64:
                    target.setFloat64(source.float64_value);
                    break;
                case ScalarKind::int64:
                    target.setInt64(source.int64_value);
                    break;
                case ScalarKind::text:
                    target.setText(source.text_value);
                    break;
                }
            }
        }

        void invoke(std::uint64_t invocation_id, std::uint32_t operation_id,
                    const TensorDescriptors &inputs,
                    const TensorDescriptors &outputs,
                    const std::vector<ScalarArgument> &scalars) {
            auto request = plugin.invokeKnownRequest();
            write_invocation(request.initInvocation(), invocation_id,
                             operation_id, inputs, outputs, scalars);
            io.provider->getTimer()
                .timeoutAfter(30 * kj::SECONDS, request.send())
                .wait(io.waitScope);
        }

        InvocationProfile
        invoke_profiled(std::uint64_t invocation_id, std::uint32_t operation_id,
                        const TensorDescriptors &inputs,
                        const TensorDescriptors &outputs,
                        const std::vector<ScalarArgument> &scalars) {
            auto request = plugin.invokeKnownProfiledRequest();
            write_invocation(request.initInvocation(), invocation_id,
                             operation_id, inputs, outputs, scalars);
            const auto rpc_started = std::chrono::steady_clock::now();
            auto response = io.provider->getTimer()
                                .timeoutAfter(30 * kj::SECONDS, request.send())
                                .wait(io.waitScope);
            const auto rpc_ns =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - rpc_started)
                    .count();
            const auto worker = response.getMetrics();
            return InvocationProfile{
                .rpc_ns = static_cast<std::uint64_t>(rpc_ns),
                .worker_input_views_ns = worker.getInputViewsNs(),
                .worker_output_views_ns = worker.getOutputViewsNs(),
                .worker_dispatch_ns = worker.getDispatchNs(),
                .worker_kernel_ns = worker.getKernelNs(),
            };
        }

        void send_control(capnp::MessageBuilder &message,
                          std::uint64_t transfer_id, int fd) {
            auto words = capnp::messageToFlatArray(message);
            auto bytes = words.asBytes();
            iovec io_vector{const_cast<kj::byte *>(bytes.begin()),
                            bytes.size()};
            msghdr header{};
            header.msg_iov = &io_vector;
            header.msg_iovlen = 1;
            std::array<char, CMSG_SPACE(sizeof(int))> ancillary{};
            if (fd >= 0) {
                header.msg_control = ancillary.data();
                header.msg_controllen = ancillary.size();
                auto *control = CMSG_FIRSTHDR(&header);
                control->cmsg_level = SOL_SOCKET;
                control->cmsg_type = SCM_RIGHTS;
                control->cmsg_len = CMSG_LEN(sizeof(int));
                std::memcpy(CMSG_DATA(control), &fd, sizeof(fd));
            }
            ssize_t sent;
            do {
                sent = ::sendmsg(control_fd.get(), &header, MSG_NOSIGNAL);
            } while (sent < 0 && errno == EINTR);
            if (sent < 0 || static_cast<std::size_t>(sent) != bytes.size()) {
                throw std::runtime_error(
                    "Failed to send buffer control message");
            }

            alignas(capnp::word) std::array<std::byte, 64 * 1024> response{};
            ssize_t received;
            do {
                received = ::recv(control_fd.get(), response.data(),
                                  response.size(), 0);
            } while (received < 0 && errno == EINTR);
            if (received <= 0 || received % sizeof(capnp::word) != 0) {
                throw std::runtime_error(
                    "Invalid buffer control acknowledgement");
            }
            auto word_array = kj::arrayPtr(
                reinterpret_cast<const capnp::word *>(response.data()),
                static_cast<std::size_t>(received) / sizeof(capnp::word));
            capnp::FlatArrayMessageReader reader(word_array);
            auto acknowledgement = reader.getRoot<::BufferTransferAck>();
            if (acknowledgement.getTransferId() != transfer_id) {
                throw std::runtime_error(
                    "Worker acknowledged an unexpected buffer request");
            }
            if (acknowledgement.which() == ::BufferTransferAck::ERROR) {
                throw std::runtime_error(
                    "Worker rejected buffer request: " +
                    std::string(acknowledgement.getError().cStr()));
            }
        }

        OwnedFd control_fd;
        kj::AsyncIoContext io;
        kj::Own<kj::AsyncIoStream> stream;
        capnp::TwoPartyClient client;
        ::Plugin::Client plugin;
    };

    struct Command {
        Command(void *callable, void (*execute)(void *, Worker &))
            : callable(callable), execute(execute) {}

        void *callable;
        void (*execute)(void *, Worker &);
        std::exception_ptr error;
        std::binary_semaphore complete{0};
    };

    Impl(int rpc_fd, int control_fd, std::uint64_t expected_fingerprint)
        : rpc_fd(rpc_fd), control_fd(control_fd),
          interrupt_rpc_fd(duplicate_fd(rpc_fd)),
          interrupt_control_fd(duplicate_fd(control_fd)),
          expected_fingerprint(expected_fingerprint),
          thread([this] { run(); }) {
        startup_complete.acquire();
        if (startup_error) {
            thread.join();
            std::rethrow_exception(startup_error);
        }
    }

    ~Impl() { close(); }

    template <typename Function> void submit(Function &&function) {
        std::unique_lock serial(submit_mutex);
        if (stopping.load(std::memory_order_acquire)) {
            throw std::runtime_error("Native worker session is closed");
        }

        using Callable = std::decay_t<Function>;
        Callable callable(std::forward<Function>(function));
        Command current(&callable, [](void *value, Worker &worker) {
            (*static_cast<Callable *>(value))(worker);
        });
        command = &current;
        command_ready.release();
        current.complete.acquire();
        command = nullptr;
        if (current.error) {
            std::rethrow_exception(current.error);
        }
    }

    void run() {
        std::optional<Worker> worker;
        try {
            worker.emplace(std::move(rpc_fd), std::move(control_fd),
                           expected_fingerprint);
        } catch (...) {
            startup_error = std::current_exception();
        }
        startup_complete.release();
        if (!worker)
            return;

        while (true) {
            command_ready.acquire();
            auto *current = command;
            if (current == nullptr) {
                return;
            }
            try {
                current->execute(current->callable, *worker);
            } catch (const kj::Exception &error) {
                try {
                    current->error = std::make_exception_ptr(kj_error(error));
                } catch (...) {
                    current->error = std::current_exception();
                }
            } catch (...) {
                current->error = std::current_exception();
            }
            current->complete.release();
        }
    }

    void close() {
        std::lock_guard closing(close_mutex);
        if (closed)
            return;
        stopping.store(true, std::memory_order_release);
        ::shutdown(interrupt_rpc_fd.get(), SHUT_RDWR);
        ::shutdown(interrupt_control_fd.get(), SHUT_RDWR);
        {
            std::lock_guard serial(submit_mutex);
            command_ready.release();
        }
        if (thread.joinable())
            thread.join();
        {
            std::lock_guard lock(mapping_mutex);
            mappings.clear();
        }
        closed = true;
    }

    void finish_invocation(std::uint64_t invocation_id) {
        std::lock_guard lock(mapping_mutex);
        std::erase_if(mappings, [invocation_id](const auto &item) {
            return !item.second.arena && item.second.writable &&
                   item.second.invocation_id == invocation_id;
        });
    }

    void abort_invocation(std::uint64_t invocation_id) {
        std::lock_guard lock(mapping_mutex);
        for (auto item = mappings.begin(); item != mappings.end();) {
            const auto &mapping = item->second;
            if (!mapping.arena && mapping.writable &&
                mapping.invocation_id == invocation_id) {
                const auto transfer_id = next_transfer_id++;
                submit([mapping, transfer_id](Worker &worker) {
                    worker.retire_buffer(mapping, transfer_id);
                });
                item = mappings.erase(item);
                ++retirements;
            } else {
                ++item;
            }
        }
    }

    OwnedFd rpc_fd;
    OwnedFd control_fd;
    OwnedFd interrupt_rpc_fd;
    OwnedFd interrupt_control_fd;
    std::uint64_t expected_fingerprint;
    std::binary_semaphore startup_complete{0};
    std::binary_semaphore command_ready{0};
    std::mutex submit_mutex;
    std::mutex close_mutex;
    std::atomic<bool> stopping{false};
    Command *command = nullptr;
    bool closed = false;
    std::exception_ptr startup_error;
    mutable std::mutex mapping_mutex;
    std::unordered_map<MappingKey, Mapping, MappingKeyHash> mappings;
    std::uint64_t next_transfer_id = 1;
    std::uint64_t transfers = 0;
    std::uint64_t retirements = 0;
    std::thread thread;
};

Session::Session(int rpc_fd, int control_fd, std::uint64_t expected_fingerprint)
    : impl_(std::make_unique<Impl>(rpc_fd, control_fd, expected_fingerprint)) {}

Session::~Session() = default;

bool Session::mapping_required(const Mapping &mapping) const {
    std::lock_guard lock(impl_->mapping_mutex);
    const auto existing =
        impl_->mappings.find(MappingKey{mapping.buffer_id, mapping.generation});
    if (existing == impl_->mappings.end())
        return true;
    if (existing->second.writable || !mapping.writable)
        return false;
    return true;
}

void Session::map_buffer(const Mapping &mapping, int fd) {
    OwnedFd owned_fd(fd);
    std::lock_guard lock(impl_->mapping_mutex);
    const MappingKey key{mapping.buffer_id, mapping.generation};
    const auto existing = impl_->mappings.find(key);
    if (existing != impl_->mappings.end()) {
        if (existing->second.writable || !mapping.writable)
            return;
        const auto transfer_id = impl_->next_transfer_id++;
        const auto previous = existing->second;
        impl_->submit([previous, transfer_id](Impl::Worker &worker) {
            worker.retire_buffer(previous, transfer_id);
        });
        impl_->mappings.erase(existing);
        ++impl_->retirements;
    }
    const auto transfer_id = impl_->next_transfer_id++;
    const auto raw_fd = owned_fd.get();
    impl_->submit([mapping, raw_fd, transfer_id](Impl::Worker &worker) {
        worker.map_buffer(mapping, duplicate_fd(raw_fd), transfer_id);
    });
    impl_->mappings.emplace(key, mapping);
    ++impl_->transfers;
}

void Session::retire_buffer(const Mapping &mapping) {
    std::lock_guard lock(impl_->mapping_mutex);
    const MappingKey key{mapping.buffer_id, mapping.generation};
    if (impl_->mappings.find(key) == impl_->mappings.end())
        return;
    const auto transfer_id = impl_->next_transfer_id++;
    impl_->submit([mapping, transfer_id](Impl::Worker &worker) {
        worker.retire_buffer(mapping, transfer_id);
    });
    impl_->mappings.erase(key);
    ++impl_->retirements;
}

void Session::abort_invocation(std::uint64_t invocation_id) {
    impl_->abort_invocation(invocation_id);
}

void Session::invoke(std::uint64_t invocation_id, std::uint32_t operation_id,
                     const TensorDescriptors &inputs,
                     const TensorDescriptors &outputs,
                     const std::vector<ScalarArgument> &scalars) {
    try {
        impl_->submit([&](Impl::Worker &worker) {
            worker.invoke(invocation_id, operation_id, inputs, outputs,
                          scalars);
        });
    } catch (...) {
        impl_->finish_invocation(invocation_id);
        throw;
    }
    impl_->finish_invocation(invocation_id);
}

InvocationProfile Session::invoke_profiled(
    std::uint64_t invocation_id, std::uint32_t operation_id,
    const TensorDescriptors &inputs, const TensorDescriptors &outputs,
    const std::vector<ScalarArgument> &scalars) {
    InvocationProfile profile;
    const auto submitted = std::chrono::steady_clock::now();
    try {
        impl_->submit([&](Impl::Worker &worker) {
            profile.queue_wait_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - submitted)
                    .count());
            const auto worker_profile = worker.invoke_profiled(
                invocation_id, operation_id, inputs, outputs, scalars);
            profile.rpc_ns = worker_profile.rpc_ns;
            profile.worker_input_views_ns =
                worker_profile.worker_input_views_ns;
            profile.worker_output_views_ns =
                worker_profile.worker_output_views_ns;
            profile.worker_dispatch_ns = worker_profile.worker_dispatch_ns;
            profile.worker_kernel_ns = worker_profile.worker_kernel_ns;
        });
    } catch (...) {
        impl_->finish_invocation(invocation_id);
        throw;
    }
    impl_->finish_invocation(invocation_id);
    return profile;
}

void Session::ping(std::uint64_t nonce) {
    impl_->submit([nonce](Impl::Worker &worker) { worker.ping(nonce); });
}

void Session::close() { impl_->close(); }

std::uint64_t Session::transfer_count() const {
    std::lock_guard lock(impl_->mapping_mutex);
    return impl_->transfers;
}

std::uint64_t Session::retirement_count() const {
    std::lock_guard lock(impl_->mapping_mutex);
    return impl_->retirements;
}

} // namespace wmfs::native
