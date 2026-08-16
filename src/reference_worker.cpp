#include "wmfs/reference/kernels.hpp"
#include "wmfs/reference/mapped_buffers.hpp"

#include <c10/core/InferenceMode.h>
#include <capnp/rpc-twoparty.h>
#include <gnu/libc-version.h>
#include <kj/async-io.h>
#include <torch/version.h>

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
        !result.has_import || result.interface_name != "ReferencePlugin") {
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

void execute_known(std::uint32_t operation_id, std::vector<TensorLease> &inputs,
                   std::vector<TensorLease> &outputs,
                   capnp::List<ScalarArgument>::Reader scalars) {
    if (operation_id == 1) {
        require(inputs.size() == 2 && outputs.size() == 1 &&
                    scalars.size() == 0,
                "Invalid matmul invocation");
        matmul_out(inputs[0].tensor(), inputs[1].tensor(), outputs[0].tensor());
        return;
    }
    if (operation_id == 2) {
        require(inputs.size() == 1 && outputs.size() == 3 &&
                    scalars.size() == 1 && scalars[0].getParameter() == 0,
                "Invalid svd invocation");
        require(scalars[0].which() == ScalarArgument::BOOLEAN,
                "Scalar argument does not match operation metadata");
        svd_out(inputs[0].tensor(), scalars[0].getBoolean(),
                outputs[0].tensor(), outputs[1].tensor(), outputs[2].tensor());
        return;
    }
    if (operation_id == 3) {
        require(inputs.size() == 1 && outputs.size() == 1 &&
                    scalars.size() == 1 && scalars[0].getParameter() == 0,
                "Invalid add_scalar invocation");
        require(scalars[0].which() == ScalarArgument::FLOAT64,
                "Scalar argument does not match operation metadata");
        add_scalar_out(inputs[0].tensor(), scalars[0].getFloat64(),
                       outputs[0].tensor());
        return;
    }
    if (operation_id == 4) {
        require(inputs.size() == 3 && outputs.size() == 2 &&
                    scalars.size() == 0,
                "Invalid matmul VJP invocation");
        matmul_vjp_out(inputs[0].tensor(), inputs[1].tensor(),
                       inputs[2].tensor(), outputs[0].tensor(),
                       outputs[1].tensor());
        return;
    }
    if (operation_id == 5) {
        require(inputs.size() == 1 && outputs.size() == 1 &&
                    scalars.size() == 0,
                "Invalid add_scalar VJP invocation");
        add_scalar_vjp_out(inputs[0].tensor(), outputs[0].tensor());
        return;
    }
    throw std::invalid_argument("Unknown operation ID " +
                                std::to_string(operation_id));
}

class ReferenceServer final : public ReferencePlugin::Server {
  public:
    explicit ReferenceServer(MappedBufferCache &buffers) : buffers_(buffers) {}

  protected:
    kj::Promise<void> getMetadata(GetMetadataContext context) override {
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

    kj::Promise<void> invokeKnown(InvokeKnownContext context) override {
        return translate_errors(
            [&] { run_known(context.getParams().getInvocation(), false); });
    }

    kj::Promise<void>
    invokeKnownProfiled(InvokeKnownProfiledContext context) override {
        return translate_errors([&] {
            auto measured =
                run_known(context.getParams().getInvocation(), true);
            auto metrics = context.getResults().initMetrics();
            metrics.setInputViewsNs(measured.input_views_ns);
            metrics.setOutputViewsNs(measured.output_views_ns);
            metrics.setDispatchNs(measured.dispatch_ns);
            metrics.setKernelNs(measured.kernel_ns);
        });
    }

  private:
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
        execute_known(invocation.getOperationId(), inputs, outputs,
                      invocation.getScalars());
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
};

} // namespace

int run_worker(int argc, char **argv) {
    auto arguments = parse_arguments(argc, argv);
    validate_socket(arguments.rpc_fd);
    validate_socket(arguments.control_fd, SOCK_SEQPACKET);

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

    try {
        auto io = kj::setupAsyncIo();
        auto stream = io.lowLevelProvider->wrapSocketFd(
            kj::AutoCloseFd(arguments.rpc_fd));
        ReferencePlugin::Client bootstrap(kj::heap<ReferenceServer>(buffers));
        capnp::TwoPartyServer server(bootstrap);
        server.accept(*stream).wait(io.waitScope);
    } catch (...) {
        ::shutdown(arguments.control_fd, SHUT_RDWR);
        receiver.join();
        throw;
    }

    ::shutdown(arguments.control_fd, SHUT_RDWR);
    receiver.join();
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
