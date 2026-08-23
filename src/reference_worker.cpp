#include "wmfs/reference/kernels.hpp"
#include "wmfs/reference/mapped_buffers.hpp"
#include "wmfs/ring.hpp"
#include "wmfs/unique_fd.hpp"

#include <ATen/ops/count_nonzero.h>
#include <c10/core/InferenceMode.h>
#include <capnp/message.h>
#include <capnp/rpc-twoparty.h>
#include <gnu/libc-version.h>
#include <kj/async-io.h>
#include <torch/version.h>

#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "wmfs-reference/reference.capnp.h"

namespace wmfs::reference {
namespace {

#define WMFS_STRINGIFY_INNER(value) #value
#define WMFS_STRINGIFY(value) WMFS_STRINGIFY_INNER(value)

struct Arguments {
    int rpc_fd = -1;
    int control_fd = -1;
    int command_ring_fd = -1;
    int command_data_fd = -1;
    int command_space_fd = -1;
    int completion_ring_fd = -1;
    int completion_data_fd = -1;
    int completion_space_fd = -1;
    std::uint64_t ring_generation = 0;
    std::string interface_name;
    bool has_schema = false;
    bool has_import = false;
};

std::uint64_t nanoseconds_since(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

std::uint64_t steady_nanoseconds() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

Arguments parse_arguments(int argc, char **argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        std::string_view argument(argv[index]);
        if (argument == "--help") {
            std::cout
                << "Usage: wmfs-reference-worker --rpc-fd FD --fd-socket-fd FD "
                   "--schema PATH --interface ReferencePlugin --schema-import "
                   "PATH\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::invalid_argument("Missing value for " +
                                        std::string(argument));
        }
        std::string_view value(argv[++index]);
        if (argument == "--rpc-fd") {
            result.rpc_fd = std::stoi(std::string(value));
        } else if (argument == "--fd-socket-fd") {
            result.control_fd = std::stoi(std::string(value));
        } else if (argument == "--command-ring-fd") {
            result.command_ring_fd = std::stoi(std::string(value));
        } else if (argument == "--command-data-fd") {
            result.command_data_fd = std::stoi(std::string(value));
        } else if (argument == "--command-space-fd") {
            result.command_space_fd = std::stoi(std::string(value));
        } else if (argument == "--completion-ring-fd") {
            result.completion_ring_fd = std::stoi(std::string(value));
        } else if (argument == "--completion-data-fd") {
            result.completion_data_fd = std::stoi(std::string(value));
        } else if (argument == "--completion-space-fd") {
            result.completion_space_fd = std::stoi(std::string(value));
        } else if (argument == "--ring-generation") {
            result.ring_generation = std::stoull(std::string(value));
        } else if (argument == "--schema") {
            result.has_schema = true;
        } else if (argument == "--interface") {
            result.interface_name = value;
        } else if (argument == "--schema-import") {
            result.has_import = true;
        } else {
            throw std::invalid_argument("Unknown argument: " +
                                        std::string(argument));
        }
    }
    if (result.rpc_fd < 0 || result.control_fd < 0 || !result.has_schema ||
        !result.has_import || result.interface_name != "ReferencePlugin" ||
        result.command_ring_fd < 0 || result.command_data_fd < 0 ||
        result.command_space_fd < 0 || result.completion_ring_fd < 0 ||
        result.completion_data_fd < 0 || result.completion_space_fd < 0 ||
        result.ring_generation == 0) {
        throw std::invalid_argument("Missing or invalid worker arguments");
    }
    if (result.rpc_fd == result.control_fd) {
        throw std::invalid_argument("RPC and buffer sockets must be distinct");
    }
    return result;
}

void validate_socket(int fd, int expected_type = 0) {
    int type = 0;
    socklen_t length = sizeof(type);
    if (::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &length) < 0) {
        throw std::invalid_argument("Worker descriptor is not an open socket");
    }
    if (expected_type != 0 && type != expected_type) {
        throw std::invalid_argument("Buffer descriptor is not SOCK_SEQPACKET");
    }
}

std::string executable_path() {
    std::array<char, 4096> buffer{};
    auto length =
        ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (length < 0) {
        return "/proc/self/exe";
    }
    return std::string(buffer.data(), static_cast<std::size_t>(length));
}

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

#include "reference_dispatch.inc"

class ReferenceServer final : public ReferencePlugin::Server {
  public:
    ReferenceServer(MappedBufferCache &buffers, std::uint32_t ring_capacity,
                    std::uint64_t ring_generation)
        : buffers_(buffers), ring_capacity_(ring_capacity),
          ring_generation_(ring_generation) {}

  protected:
    kj::Promise<void> getMetadata(GetMetadataContext context) override {
        if (PLUGIN_METADATA.get().getFingerprint() !=
            WMFS_METADATA_FINGERPRINT) {
            throw std::logic_error(
                "Reference schema and generated dispatch fingerprints differ");
        }
        context.getResults().setMetadata(PLUGIN_METADATA.get());
        return kj::READY_NOW;
    }

    kj::Promise<void>
    getProtocolVersion(GetProtocolVersionContext context) override {
        context.getResults().setVersion(PROTOCOL_VERSION);
        return kj::READY_NOW;
    }

    kj::Promise<void> ping(PingContext context) override {
        context.getResults().setNonce(context.getParams().getNonce());
        return kj::READY_NOW;
    }

    kj::Promise<void> getEnvironment(GetEnvironmentContext context) override {
        auto environment = context.getResults().initEnvironment();
        environment.setPythonVersion("none");
        environment.setTorchVersion(
            WMFS_STRINGIFY(TORCH_VERSION_MAJOR) "." WMFS_STRINGIFY(
                TORCH_VERSION_MINOR) "." WMFS_STRINGIFY(TORCH_VERSION_PATCH));
        environment.setGlibcVersion(gnu_get_libc_version());
        environment.setExecutable(executable_path());
        return kj::READY_NOW;
    }

    kj::Promise<void>
    getRingHandshake(GetRingHandshakeContext context) override {
        auto ring = context.getResults().initRing();
        ring.setAbiMajor(WMFS_RING_ABI_MAJOR);
        ring.setAbiMinor(WMFS_RING_ABI_MINOR);
        ring.setHeaderSize(WMFS_RING_HEADER_SIZE);
        ring.setRecordSize(WMFS_RING_RECORD_SIZE);
        ring.setCapacity(ring_capacity_);
        ring.setGeneration(ring_generation_);
        ring.setCapabilities(3);
        return kj::READY_NOW;
    }

    kj::Promise<void> invokeKnown(InvokeKnownContext context) override {
        return translate_errors([&] {
            auto outcome = context.getResults().initOutcome();
            try {
                run_known(context.getParams().getInvocation(), false);
                outcome.setSuccess();
            } catch (const OperationFailure &error) {
                auto result = outcome.initOperationError();
                result.setType(error.type);
                result.setMessage(error.what());
            }
        });
    }

    kj::Promise<void>
    invokeKnownProfiled(InvokeKnownProfiledContext context) override {
        return translate_errors([&] {
            auto outcome = context.getResults().initOutcome();
            try {
                auto measured =
                    run_known(context.getParams().getInvocation(), true);
                outcome.setSuccess();
                auto metrics = context.getResults().initMetrics();
                metrics.setInputViewsNs(measured.input_views_ns);
                metrics.setOutputViewsNs(measured.output_views_ns);
                metrics.setDispatchNs(measured.dispatch_ns);
                metrics.setKernelNs(measured.kernel_ns);
            } catch (const OperationFailure &error) {
                auto result = outcome.initOperationError();
                result.setType(error.type);
                result.setMessage(error.what());
            }
        });
    }

    kj::Promise<void> planOutputs(PlanOutputsContext context) override {
        return translate_errors([&] {
            auto invocation = context.getParams().getInvocation();
            require(invocation.getOperationId() == 6,
                    "Operation has no dynamic output planner");
            require(invocation.getInputs().size() == 1,
                    "Output planning has an invalid input count");
            require(invocation.getScalars().size() == 1 &&
                        invocation.getScalars()[0].isInt64() &&
                        invocation.getScalars()[0].getInt64() >= 0 &&
                        invocation.getScalars()[0].getInt64() <= 1,
                    "Output planning has an invalid enum scalar");
            auto invocation_id = invocation.getInvocationId();
            auto input =
                buffers_.tensor(invocation.getInputs()[0], invocation_id);
            auto outcome = context.getResults().initOutcome();
            std::int64_t count;
            try {
                count = at::count_nonzero(input.tensor()).item<std::int64_t>();
                if (count == 0)
                    throw std::invalid_argument(
                        "nonzero does not yet support an empty result");
            } catch (const OperationFailure &error) {
                auto result = outcome.initOperationError();
                result.setType(error.type);
                result.setMessage(error.what());
                return;
            } catch (const c10::Error &error) {
                auto result = outcome.initOperationError();
                result.setType("RuntimeError");
                result.setMessage(error.what_without_backtrace());
                return;
            }
            outcome.setSuccess();
            auto outputs = context.getResults().initOutputs(1);
            outputs[0].setOutput(0);
            auto shape = outputs[0].initShape(2);
            shape.set(0, static_cast<std::uint64_t>(count));
            shape.set(1, static_cast<std::uint64_t>(input.tensor().dim()));
            outputs[0].setDtype(DType::INT64);
        });
    }

  private:
    struct OperationFailure : std::runtime_error {
        std::string type;

        OperationFailure(std::string type, std::string message)
            : std::runtime_error(std::move(message)), type(std::move(type)) {}
    };

    struct InvocationMeasurements {
        std::uint64_t input_views_ns{};
        std::uint64_t output_views_ns{};
        std::uint64_t dispatch_ns{};
        std::uint64_t kernel_ns{};
    };

    InvocationMeasurements run_known(KnownInvocation::Reader invocation,
                                     bool profiled) {
        auto invocation_id = invocation.getInvocationId();
        struct InvocationCleanup {
            MappedBufferCache &buffers;
            std::uint64_t invocation_id;
            ~InvocationCleanup() { buffers.finish_invocation(invocation_id); }
        } cleanup{buffers_, invocation_id};

        c10::InferenceMode inference_mode;
        auto started = profiled ? std::chrono::steady_clock::now()
                                : std::chrono::steady_clock::time_point{};
        auto view_started = profiled ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
        std::vector<TensorLease> inputs;
        inputs.reserve(invocation.getInputs().size());
        for (auto descriptor : invocation.getInputs()) {
            inputs.push_back(buffers_.tensor(descriptor, invocation_id));
        }
        auto input_views_ns = profiled ? nanoseconds_since(view_started) : 0;

        if (profiled) {
            view_started = std::chrono::steady_clock::now();
        }
        std::vector<TensorLease> outputs;
        outputs.reserve(invocation.getOutputs().size());
        for (auto descriptor : invocation.getOutputs()) {
            outputs.push_back(buffers_.tensor(descriptor, invocation_id, true));
        }
        auto output_views_ns = profiled ? nanoseconds_since(view_started) : 0;

        auto kernel_started = profiled
                                  ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
        try {
            execute_known(invocation.getOperationId(), inputs, outputs,
                          invocation.getScalars());
        } catch (const c10::Error &error) {
            throw OperationFailure("RuntimeError",
                                   error.what_without_backtrace());
        } catch (const std::invalid_argument &error) {
            throw OperationFailure("ValueError", error.what());
        } catch (const std::exception &error) {
            throw OperationFailure("RuntimeError", error.what());
        }
        auto kernel_ns = profiled ? nanoseconds_since(kernel_started) : 0;
        auto elapsed_ns = profiled ? nanoseconds_since(started) : 0;
        auto measured_ns = input_views_ns + output_views_ns + kernel_ns;
        return {
            input_views_ns,
            output_views_ns,
            elapsed_ns > measured_ns ? elapsed_ns - measured_ns : 0,
            kernel_ns,
        };
    }

    template <typename Function>
    static kj::Promise<void> translate_errors(Function &&function) {
        try {
            std::forward<Function>(function)();
            return kj::READY_NOW;
        } catch (const c10::Error &error) {
            return kj::Promise<void>(
                KJ_EXCEPTION(FAILED, error.what_without_backtrace()));
        } catch (const std::exception &error) {
            return kj::Promise<void>(KJ_EXCEPTION(FAILED, error.what()));
        }
    }

    MappedBufferCache &buffers_;
    std::uint32_t ring_capacity_;
    std::uint64_t ring_generation_;
};

DType dtype_to_capnp(std::uint32_t dtype) {
    switch (dtype) {
    case WMFS_RING_DTYPE_FLOAT32:
        return DType::FLOAT32;
    case WMFS_RING_DTYPE_FLOAT64:
        return DType::FLOAT64;
    case WMFS_RING_DTYPE_INT64:
        return DType::INT64;
    case WMFS_RING_DTYPE_UINT8:
        return DType::UINT8;
    default:
        throw std::invalid_argument("Unsupported ring tensor dtype");
    }
}

TensorLease ring_tensor(MappedBufferCache &buffers,
                        const wmfs_ring_tensor_descriptor_v1 &source,
                        std::uint64_t invocation_id, bool writable) {
    capnp::MallocMessageBuilder message;
    auto descriptor = message.initRoot<TensorDescriptor>();
    descriptor.setBufferId(source.buffer_id);
    descriptor.setGeneration(
        static_cast<std::uint32_t>(source.buffer_generation));
    descriptor.setAllocationId(source.allocation_id);
    descriptor.setOffset(source.byte_offset);
    descriptor.setByteLength(source.byte_length);
    descriptor.setDtype(dtype_to_capnp(source.dtype));
    auto shape = descriptor.initShape(source.rank);
    auto strides = descriptor.initStrides(source.rank);
    for (std::uint16_t index = 0; index < source.rank; ++index) {
        require(source.shape[index] > 0, "Ring tensor shape is invalid");
        shape.set(index, static_cast<std::uint64_t>(source.shape[index]));
        strides.set(index, source.strides[index]);
    }
    return buffers.tensor(descriptor.asReader(), invocation_id, writable);
}

void execute_ring(const wmfs_ring_record_v1 &command,
                  std::vector<TensorLease> &inputs,
                  std::vector<TensorLease> &outputs) {
    switch (command.operation_id) {
    case 1:
        require(inputs.size() == 2 && outputs.size() == 1 &&
                    command.scalar_count == 0,
                "Invalid matmul invocation");
        matmul_out(inputs[0].tensor(), inputs[1].tensor(), outputs[0].tensor());
        break;
    case 2:
        require(inputs.size() == 1 && outputs.size() == 3 &&
                    command.scalar_count == 1 &&
                    command.scalars[0].kind == WMFS_RING_SCALAR_BOOLEAN &&
                    command.scalars[0].parameter_index == 0,
                "Invalid svd invocation");
        svd_out(inputs[0].tensor(), command.scalars[0].bits != 0,
                outputs[0].tensor(), outputs[1].tensor(), outputs[2].tensor());
        break;
    case 3: {
        require(inputs.size() == 1 && outputs.size() == 1 &&
                    command.scalar_count == 1 &&
                    command.scalars[0].kind == WMFS_RING_SCALAR_FLOAT64,
                "Invalid add_scalar invocation");
        double value;
        std::memcpy(&value, &command.scalars[0].bits, sizeof(value));
        add_scalar_out(inputs[0].tensor(), value, outputs[0].tensor());
        break;
    }
    case 4:
        require(inputs.size() == 3 && outputs.size() == 2,
                "Invalid matmul_vjp invocation");
        matmul_vjp_out(inputs[0].tensor(), inputs[1].tensor(),
                       inputs[2].tensor(), outputs[0].tensor(),
                       outputs[1].tensor());
        break;
    case 5:
        require(inputs.size() == 1 && outputs.size() == 1,
                "Invalid add_scalar_vjp invocation");
        add_scalar_vjp_out(inputs[0].tensor(), outputs[0].tensor());
        break;
    case 6:
        require(inputs.size() == 1 && outputs.size() == 1 &&
                    command.scalar_count == 1 &&
                    command.scalars[0].kind == WMFS_RING_SCALAR_INT64 &&
                    command.scalars[0].bits <= 1,
                "Invalid nonzero invocation");
        nonzero_out(inputs[0].tensor(),
                    static_cast<std::int64_t>(command.scalars[0].bits),
                    outputs[0].tensor());
        break;
    default:
        throw std::invalid_argument("Unknown operation ID");
    }
}

void set_error(wmfs_ring_record_v1 &completion, std::uint32_t status,
               std::string_view type, std::string_view message) {
    completion.status = status;
    const auto type_size = std::min(type.size(), sizeof(completion.error.type));
    const auto message_size =
        std::min(message.size(), sizeof(completion.error.message));
    completion.error.type_length = static_cast<std::uint32_t>(type_size);
    completion.error.message_length = static_cast<std::uint32_t>(message_size);
    completion.error.flags =
        (type_size != type.size() ? WMFS_RING_ERROR_FLAG_TRUNCATED_TYPE : 0) |
        (message_size != message.size() ? WMFS_RING_ERROR_FLAG_TRUNCATED_MESSAGE
                                        : 0);
    std::memcpy(completion.error.type, type.data(), type_size);
    std::memcpy(completion.error.message, message.data(), message_size);
}

void run_ring(wmfs::RingConsumer commands, wmfs::RingProducer completions,
              MappedBufferCache &buffers) {
    for (;;) {
        wmfs_ring_record_v1 command{};
        if (commands.pop(command) != wmfs::RingWaitResult::success)
            return;
        const auto profiled =
            (command.flags & WMFS_RING_RECORD_FLAG_PROFILE) != 0;
        const auto worker_dequeued_ns = profiled ? steady_nanoseconds() : 0;
        wmfs_ring_record_v1 completion{};
        completion.kind = command.kind == WMFS_RING_COMMAND_INVOKE
                              ? WMFS_RING_COMPLETION_INVOKE
                          : command.kind == WMFS_RING_COMMAND_PLAN_OUTPUTS
                              ? WMFS_RING_COMPLETION_PLAN_OUTPUTS
                              : WMFS_RING_COMPLETION_PONG;
        completion.flags = command.flags;
        completion.session_generation = command.session_generation;
        completion.submission_id = command.submission_id;
        completion.invocation_id = command.invocation_id;
        completion.operation_id = command.operation_id;
        completion.status = WMFS_RING_STATUS_OK;
        completion.profile.command_published_ns =
            command.profile.command_published_ns;
        completion.profile.worker_dequeued_ns = worker_dequeued_ns;
        completion.profile.worker_started_ns =
            profiled ? steady_nanoseconds() : 0;
        try {
            c10::InferenceMode inference_mode;
            std::vector<TensorLease> inputs;
            std::vector<TensorLease> outputs;
            const auto started = std::chrono::steady_clock::now();
            for (std::uint16_t index = 0; index < command.tensor_count;
                 ++index) {
                const auto &descriptor = command.tensors[index];
                const auto view_started = std::chrono::steady_clock::now();
                if (descriptor.kind == WMFS_RING_TENSOR_INPUT) {
                    inputs.push_back(
                        ring_tensor(buffers, descriptor, command.invocation_id,
                                    (descriptor.flags &
                                     WMFS_RING_TENSOR_FLAG_WRITABLE) != 0));
                    completion.profile.worker_input_views_ns +=
                        nanoseconds_since(view_started);
                } else if (descriptor.kind == WMFS_RING_TENSOR_OUTPUT) {
                    outputs.push_back(ring_tensor(buffers, descriptor,
                                                  command.invocation_id, true));
                    completion.profile.worker_output_views_ns +=
                        nanoseconds_since(view_started);
                } else {
                    throw std::invalid_argument("Invalid ring tensor kind");
                }
            }
            if (command.kind == WMFS_RING_COMMAND_PLAN_OUTPUTS) {
                require(command.operation_id == 6 && inputs.size() == 1 &&
                            command.scalar_count == 1 &&
                            command.scalars[0].kind == WMFS_RING_SCALAR_INT64 &&
                            command.scalars[0].bits <= 1,
                        "Operation has no dynamic output planner");
                const auto count =
                    at::count_nonzero(inputs[0].tensor()).item<std::int64_t>();
                if (count == 0)
                    throw std::invalid_argument(
                        "nonzero does not yet support an empty result");
                completion.planned_output_count = 1;
                completion.planned_outputs[0].dtype = WMFS_RING_DTYPE_INT64;
                completion.planned_outputs[0].rank = 2;
                completion.planned_outputs[0].output_index = 0;
                completion.planned_outputs[0].shape[0] = count;
                completion.planned_outputs[0].shape[1] =
                    inputs[0].tensor().dim();
            } else if (command.kind == WMFS_RING_COMMAND_INVOKE) {
                const auto kernel = std::chrono::steady_clock::now();
                execute_ring(command, inputs, outputs);
                completion.profile.worker_kernel_ns = nanoseconds_since(kernel);
                const auto elapsed = nanoseconds_since(started);
                completion.profile.worker_dispatch_ns =
                    elapsed > completion.profile.worker_kernel_ns
                        ? elapsed - completion.profile.worker_kernel_ns
                        : 0;
            } else if (command.kind == WMFS_RING_COMMAND_PING) {
                const auto kernel = std::chrono::steady_clock::now();
                if (command.operation_id != 0)
                    std::this_thread::sleep_for(
                        std::chrono::nanoseconds(command.operation_id));
                completion.profile.worker_kernel_ns =
                    profiled ? nanoseconds_since(kernel) : 0;
            } else {
                throw std::invalid_argument("Unsupported ring command");
            }
        } catch (const c10::Error &error) {
            set_error(completion, WMFS_RING_STATUS_OPERATION_ERROR,
                      "RuntimeError", error.what_without_backtrace());
        } catch (const std::invalid_argument &error) {
            set_error(completion, WMFS_RING_STATUS_OPERATION_ERROR,
                      "ValueError", error.what());
        } catch (const std::exception &error) {
            set_error(completion, WMFS_RING_STATUS_INTERNAL_ERROR,
                      "RuntimeError", error.what());
        }
        buffers.finish_invocation(command.invocation_id);
        completion.profile.completion_published_ns =
            profiled ? steady_nanoseconds() : 0;
        if (completions.push(completion) != wmfs::RingWaitResult::success)
            return;
    }
}

} // namespace

int run_worker(int argc, char **argv) {
    auto arguments = parse_arguments(argc, argv);
    validate_socket(arguments.rpc_fd);
    validate_socket(arguments.control_fd, SOCK_SEQPACKET);

    auto commands = wmfs::RingConsumer::borrow(
        wmfs::UniqueFd(arguments.command_ring_fd),
        wmfs::UniqueFd(arguments.command_data_fd),
        wmfs::UniqueFd(arguments.command_space_fd), arguments.ring_generation);
    auto command_interrupt = wmfs::RingProducer::borrow(
        commands.duplicate_ring_fd(), commands.duplicate_data_event_fd(),
        commands.duplicate_space_event_fd(), arguments.ring_generation);
    const auto ring_capacity = [&] {
        struct stat status{};
        if (::fstat(commands.ring_fd(), &status) < 0)
            throw std::runtime_error("Cannot inspect command ring");
        return static_cast<std::uint32_t>(
            (status.st_size - WMFS_RING_HEADER_SIZE) / WMFS_RING_RECORD_SIZE);
    }();
    auto completions = wmfs::RingProducer::borrow(
        wmfs::UniqueFd(arguments.completion_ring_fd),
        wmfs::UniqueFd(arguments.completion_data_fd),
        wmfs::UniqueFd(arguments.completion_space_fd),
        arguments.ring_generation);

    MappedBufferCache buffers;
    std::exception_ptr receiver_error;
    std::thread receiver([&] {
        try {
            receive_buffer_transfers(arguments.control_fd, buffers);
        } catch (...) {
            receiver_error = std::current_exception();
            ::shutdown(arguments.rpc_fd, SHUT_RDWR);
        }
    });
    std::thread ring_worker([&] {
        try {
            run_ring(std::move(commands), std::move(completions), buffers);
        } catch (...) {
            ::shutdown(arguments.rpc_fd, SHUT_RDWR);
        }
    });

    try {
        auto io = kj::setupAsyncIo();
        auto stream = io.lowLevelProvider->wrapSocketFd(
            kj::AutoCloseFd(arguments.rpc_fd));
        ReferencePlugin::Client bootstrap(kj::heap<ReferenceServer>(
            buffers, ring_capacity, arguments.ring_generation));
        capnp::TwoPartyServer server(bootstrap);
        server.accept(*stream).wait(io.waitScope);
    } catch (...) {
        ::shutdown(arguments.control_fd, SHUT_RDWR);
        command_interrupt.close();
        receiver.join();
        ring_worker.join();
        throw;
    }

    ::shutdown(arguments.control_fd, SHUT_RDWR);
    command_interrupt.close();
    receiver.join();
    ring_worker.join();
    if (receiver_error) {
        std::rethrow_exception(receiver_error);
    }
    return 0;
}

} // namespace wmfs::reference

int main(int argc, char **argv) {
    try {
        return wmfs::reference::run_worker(argc, argv);
    } catch (const kj::Exception &error) {
        std::cerr << "wmfs-reference-worker: " << error.getDescription().cStr()
                  << '\n';
    } catch (const std::exception &error) {
        std::cerr << "wmfs-reference-worker: " << error.what() << '\n';
    }
    return 1;
}
