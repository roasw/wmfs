#include "wmfs/native/session.hpp"

#include <capnp/message.h>
#include <capnp/rpc-twoparty.h>
#include <capnp/serialize.h>
#include <kj/async-io.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
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

::DType dtype_from_name(const std::string &name) {
    if (name == "float32")
        return ::DType::FLOAT32;
    if (name == "float64")
        return ::DType::FLOAT64;
    if (name == "int64")
        return ::DType::INT64;
    if (name == "uint8")
        return ::DType::UINT8;
    throw std::invalid_argument("Unsupported tensor dtype: " + name);
}

void write_descriptor(::TensorDescriptor::Builder target,
                      const TensorDescriptor &source) {
    target.setBufferId(source.buffer_id);
    target.setGeneration(source.generation);
    target.setAllocationId(source.allocation_id);
    target.setOffset(source.offset);
    target.setByteLength(source.byte_length);
    target.setDtype(dtype_from_name(source.dtype));
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
        explicit Worker(int rpc_fd, int control_fd,
                        std::uint64_t expected_fingerprint)
            : io(kj::setupAsyncIo()), stream(io.lowLevelProvider->wrapSocketFd(
                                          kj::AutoCloseFd(rpc_fd))),
              client(*stream), plugin(client.bootstrap().castAs<::Plugin>()),
              control_fd(control_fd) {
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

        InvocationProfile invoke(std::uint64_t invocation_id,
                                 std::uint32_t operation_id,
                                 const std::vector<TensorDescriptor> &inputs,
                                 const std::vector<TensorDescriptor> &outputs,
                                 const std::vector<ScalarArgument> &scalars,
                                 bool profiled) {
            auto request = plugin.invokeKnownRequest();
            auto invocation = request.initInvocation();
            invocation.setInvocationId(invocation_id);
            invocation.setOperationId(operation_id);
            invocation.setProfiled(profiled);
            auto input_builders = invocation.initInputs(inputs.size());
            for (std::size_t index = 0; index < inputs.size(); ++index) {
                write_descriptor(input_builders[index], inputs[index]);
            }
            auto output_builders = invocation.initOutputs(outputs.size());
            for (std::size_t index = 0; index < outputs.size(); ++index) {
                write_descriptor(output_builders[index], outputs[index]);
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
            const auto rpc_started =
                profiled ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{};
            auto response = io.provider->getTimer()
                                .timeoutAfter(30 * kj::SECONDS, request.send())
                                .wait(io.waitScope);
            const auto rpc_ns =
                profiled ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - rpc_started)
                               .count()
                         : 0;
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

        kj::AsyncIoContext io;
        kj::Own<kj::AsyncIoStream> stream;
        capnp::TwoPartyClient client;
        ::Plugin::Client plugin;
        OwnedFd control_fd;
    };

    Impl(int rpc_fd, int control_fd, std::uint64_t expected_fingerprint)
        : rpc_fd(rpc_fd), control_fd(control_fd),
          interrupt_rpc_fd(duplicate_fd(rpc_fd)),
          interrupt_control_fd(duplicate_fd(control_fd)),
          expected_fingerprint(expected_fingerprint),
          thread([this] { run(); }) {
        std::unique_lock lock(queue_mutex);
        ready_condition.wait(lock, [this] { return ready; });
        if (startup_error) {
            lock.unlock();
            thread.join();
            std::rethrow_exception(startup_error);
        }
    }

    ~Impl() { close(); }

    template <typename Function> void submit(Function &&function) {
        auto completion = std::make_shared<std::promise<void>>();
        auto result = completion->get_future();
        {
            std::lock_guard lock(queue_mutex);
            if (stopping) {
                throw std::runtime_error("Native worker session is closed");
            }
            queue.emplace_back([function = std::forward<Function>(function),
                                completion](Worker &worker) {
                try {
                    function(worker);
                    completion->set_value();
                } catch (const kj::Exception &error) {
                    completion->set_exception(
                        std::make_exception_ptr(kj_error(error)));
                } catch (...) {
                    completion->set_exception(std::current_exception());
                }
            });
        }
        queue_condition.notify_one();
        result.get();
    }

    void run() {
        std::optional<Worker> worker;
        try {
            worker.emplace(rpc_fd.release(), control_fd.release(),
                           expected_fingerprint);
        } catch (...) {
            startup_error = std::current_exception();
        }
        {
            std::lock_guard lock(queue_mutex);
            ready = true;
        }
        ready_condition.notify_one();
        if (!worker)
            return;

        while (true) {
            std::function<void(Worker &)> command;
            {
                std::unique_lock lock(queue_mutex);
                queue_condition.wait(
                    lock, [this] { return stopping || !queue.empty(); });
                if (stopping && queue.empty())
                    break;
                command = std::move(queue.front());
                queue.pop_front();
            }
            command(*worker);
        }
    }

    void close() {
        {
            std::unique_lock lock(queue_mutex);
            if (stopping) {
                closed_condition.wait(lock, [this] { return closed; });
                return;
            }
            stopping = true;
        }
        ::shutdown(interrupt_rpc_fd.get(), SHUT_RDWR);
        ::shutdown(interrupt_control_fd.get(), SHUT_RDWR);
        queue_condition.notify_one();
        if (thread.joinable())
            thread.join();
        {
            std::lock_guard lock(mapping_mutex);
            mappings.clear();
        }
        {
            std::lock_guard lock(queue_mutex);
            closed = true;
        }
        closed_condition.notify_all();
    }

    OwnedFd rpc_fd;
    OwnedFd control_fd;
    OwnedFd interrupt_rpc_fd;
    OwnedFd interrupt_control_fd;
    std::uint64_t expected_fingerprint;
    std::thread thread;
    std::mutex queue_mutex;
    std::condition_variable queue_condition;
    std::condition_variable ready_condition;
    std::condition_variable closed_condition;
    std::deque<std::function<void(Worker &)>> queue;
    bool ready = false;
    bool stopping = false;
    bool closed = false;
    std::exception_ptr startup_error;
    mutable std::mutex mapping_mutex;
    std::unordered_map<MappingKey, Mapping, MappingKeyHash> mappings;
    std::uint64_t next_transfer_id = 1;
    std::uint64_t transfers = 0;
    std::uint64_t retirements = 0;
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
    throw std::runtime_error(
        "Cannot upgrade an existing read-only worker mapping");
}

void Session::map_buffer(const Mapping &mapping, int fd) {
    OwnedFd owned_fd(fd);
    std::lock_guard lock(impl_->mapping_mutex);
    const MappingKey key{mapping.buffer_id, mapping.generation};
    const auto existing = impl_->mappings.find(key);
    if (existing != impl_->mappings.end()) {
        if (existing->second.writable || !mapping.writable)
            return;
        throw std::runtime_error(
            "Cannot upgrade an existing read-only worker mapping");
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

InvocationProfile Session::invoke(std::uint64_t invocation_id,
                                  std::uint32_t operation_id,
                                  const std::vector<TensorDescriptor> &inputs,
                                  const std::vector<TensorDescriptor> &outputs,
                                  const std::vector<ScalarArgument> &scalars,
                                  bool profiled) {
    InvocationProfile profile;
    const auto submitted = profiled ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    try {
        impl_->submit([&](Impl::Worker &worker) {
            if (profiled) {
                profile.queue_wait_ns = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - submitted)
                        .count());
            }
            const auto worker_profile =
                worker.invoke(invocation_id, operation_id, inputs, outputs,
                              scalars, profiled);
            profile.rpc_ns = worker_profile.rpc_ns;
            profile.worker_input_views_ns =
                worker_profile.worker_input_views_ns;
            profile.worker_output_views_ns =
                worker_profile.worker_output_views_ns;
            profile.worker_dispatch_ns = worker_profile.worker_dispatch_ns;
            profile.worker_kernel_ns = worker_profile.worker_kernel_ns;
        });
    } catch (...) {
        std::lock_guard lock(impl_->mapping_mutex);
        std::erase_if(impl_->mappings, [invocation_id](const auto &item) {
            return !item.second.arena && item.second.writable &&
                   item.second.invocation_id == invocation_id;
        });
        throw;
    }
    std::lock_guard lock(impl_->mapping_mutex);
    std::erase_if(impl_->mappings, [invocation_id](const auto &item) {
        return !item.second.arena && item.second.writable &&
               item.second.invocation_id == invocation_id;
    });
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
