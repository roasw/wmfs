#include "wmfs/reference/kernels.hpp"
#include "wmfs/reference/mapped_buffers.hpp"
#include "wmfs/ring.hpp"
#include "wmfs/unique_fd.hpp"
#include <wmfs/protocol/control.h>
#include <wmfs/protocol/log.h>
#include <wmfs/reference_plugin.hpp>

#include <ATen/ops/count_nonzero.h>
#include <c10/core/InferenceMode.h>
#include <gnu/libc-version.h>
#include <torch/version.h>

#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace wmfs::reference {
namespace {

#define WMFS_STRINGIFY_INNER(value) #value
#define WMFS_STRINGIFY(value) WMFS_STRINGIFY_INNER(value)

struct StartupResources {
    UniqueFd bootstrap;
    UniqueFd fd_control;
    std::array<UniqueFd, 3> commands;
    std::array<UniqueFd, 3> completions;
    std::uint64_t generation{};
    const wmfs_plugin_api_v1 *api{};
    bool initialized{};
    struct LogSink;
    std::unique_ptr<LogSink> log;
};

struct StartupResources::LogSink {
    struct Item {
        std::uint32_t level;
        std::vector<std::uint8_t> bytes;
    };
    UniqueFd fd;
    std::uint32_t mode{};
    std::uint32_t level{20};
    std::size_t record_bytes{16384};
    std::uint64_t generation{};
    std::uint64_t sequence{};
    std::size_t capacity{256};
    std::uint64_t dropped{};
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<Item> queue;
    bool closing{};
    std::thread sender;

    ~LogSink() { close(); }
    void start() {
        sender = std::thread([this] { run(); });
    }
    void enqueue(std::uint32_t level, std::vector<std::uint8_t> bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        if (closing)
            return;
        if (queue.size() == capacity) {
            auto lowest =
                std::min_element(queue.begin(), queue.end(),
                                 [](const Item &left, const Item &right) {
                                     return left.level < right.level;
                                 });
            if (lowest != queue.end() && lowest->level <= level) {
                queue.erase(lowest);
                queue.push_back({level, std::move(bytes)});
            }
            ++dropped;
        } else
            queue.push_back({level, std::move(bytes)});
        condition.notify_one();
    }
    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            closing = true;
            condition.notify_all();
        }
        if (sender.joinable())
            sender.join();
    }
    void run() {
        for (;;) {
            Item item;
            std::uint64_t dropped_before{};
            {
                std::unique_lock<std::mutex> lock(mutex);
                condition.wait(lock,
                               [this] { return closing || !queue.empty(); });
                if (queue.empty())
                    return;
                item = std::move(queue.front());
                queue.pop_front();
                dropped_before = dropped;
                dropped = 0;
            }
            if (mode == WMFS_CONTROL_LOG_CENTRALIZED) {
                if (dropped_before &&
                    item.bytes.size() >= WMFS_LOG_HEADER_SIZE) {
                    auto synthetic = item.bytes;
                    synthetic[20] |= WMFS_LOG_RECORD_SYNTHETIC;
                    synthetic[24] = WMFS_LOG_WARNING;
                    for (unsigned index = 0; index < 8; ++index)
                        synthetic[92 + index] = static_cast<std::uint8_t>(
                            dropped_before >> (index * 8));
                    (void)::send(fd.get(), synthetic.data(), synthetic.size(),
                                 MSG_DONTWAIT | MSG_NOSIGNAL);
                }
                (void)::send(fd.get(), item.bytes.data(), item.bytes.size(),
                             MSG_DONTWAIT | MSG_NOSIGNAL);
                continue;
            }
            if (dropped_before) {
                const auto warning =
                    std::string("{\"level\":30,\"message\":\"dropped ") +
                    std::to_string(dropped_before) +
                    " plugin log records\",\"category\":\"wmfs.logging\","
                    "\"fields\":{},\"droppedBefore\":" +
                    std::to_string(dropped_before) + "}\n";
                const auto ignored =
                    ::write(fd.get(), warning.data(), warning.size());
                (void)ignored;
            }
            std::size_t offset = 0;
            while (offset < item.bytes.size()) {
                const auto written =
                    ::write(fd.get(), item.bytes.data() + offset,
                            item.bytes.size() - offset);
                if (written > 0)
                    offset += static_cast<std::size_t>(written);
                else if (written < 0 && errno == EINTR)
                    continue;
                else
                    break;
            }
        }
    }
};

std::uint64_t steady_nanoseconds();
thread_local std::uint64_t log_submission_id{};
thread_local std::uint64_t log_invocation_id{};
thread_local std::uint64_t log_operation_id{};

void store32(std::uint8_t *p, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index)
        p[index] = static_cast<std::uint8_t>(value >> (index * 8));
}
void store64(std::uint8_t *p, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index)
        p[index] = static_cast<std::uint8_t>(value >> (index * 8));
}

std::size_t utf8_prefix(const char *data, std::size_t size, std::size_t limit) {
    if (size <= limit)
        return size;
    size = limit;
    while (size && (static_cast<unsigned char>(data[size]) & 0xc0) == 0x80)
        --size;
    return size;
}

std::string json_escape(const char *data, std::size_t size) {
    std::string result;
    for (std::size_t index = 0; index < size; ++index) {
        const unsigned char value = static_cast<unsigned char>(data[index]);
        if (value == '"' || value == '\\') {
            result.push_back('\\');
            result.push_back(static_cast<char>(value));
        } else if (value >= 0x20)
            result.push_back(static_cast<char>(value));
    }
    return result;
}

std::string json_fields(const wmfs_log_field_v1 *fields,
                        std::uint32_t field_count) {
    std::string result{"{"};
    bool first = true;
    for (std::uint32_t index = 0; fields && index < field_count; ++index) {
        const auto &field = fields[index];
        if (field.struct_size < sizeof(wmfs_log_field_v1) || !field.name.data ||
            !field.name.size || field.kind < WMFS_LOG_FIELD_BOOLEAN ||
            field.kind > WMFS_LOG_FIELD_TEXT)
            continue;
        result += first ? "\"" : ",\"";
        first = false;
        result += json_escape(field.name.data, field.name.size) + "\":";
        if (field.kind == WMFS_LOG_FIELD_BOOLEAN)
            result += field.bits ? "true" : "false";
        else if (field.kind == WMFS_LOG_FIELD_INT64)
            result += std::to_string(static_cast<std::int64_t>(field.bits));
        else if (field.kind == WMFS_LOG_FIELD_UINT64)
            result += std::to_string(field.bits);
        else if (field.kind == WMFS_LOG_FIELD_FLOAT64) {
            double value{};
            std::memcpy(&value, &field.bits, sizeof(value));
            result += std::to_string(value);
        } else
            result += "\"" +
                      json_escape(field.text.data ? field.text.data : "",
                                  field.text.data ? field.text.size : 0) +
                      "\"";
    }
    return result + "}";
}

std::uint8_t log_enabled(void *context, std::uint32_t level) {
    const auto *sink = static_cast<StartupResources::LogSink *>(context);
    return sink && level >= sink->level && level >= 10 && level <= 50 &&
           level % 10 == 0;
}

void log_write(void *context, std::uint32_t level, wmfs_text_view_v1 message,
               wmfs_text_view_v1 category, const wmfs_log_field_v1 *fields,
               std::uint32_t field_count) {
    auto *sink = static_cast<StartupResources::LogSink *>(context);
    if (!log_enabled(context, level) || (!message.data && message.size) ||
        (!category.data && category.size))
        return;
    try {
        std::uint64_t sequence;
        {
            std::lock_guard<std::mutex> lock(sink->mutex);
            sequence = ++sink->sequence;
        }
        if (sink->mode == WMFS_CONTROL_LOG_WORKER_FILE) {
            const auto document =
                std::string("{\"timeNs\":") +
                std::to_string(steady_nanoseconds()) +
                ",\"sequence\":" + std::to_string(sequence) +
                ",\"level\":" + std::to_string(level) + ",\"message\":\"" +
                json_escape(message.data, message.size) + "\",\"category\":\"" +
                json_escape(category.data, category.size) +
                "\",\"fields\":" + json_fields(fields, field_count) +
                ",\"sessionId\":" + std::to_string(sink->generation) +
                ",\"submissionId\":" + std::to_string(log_submission_id) +
                ",\"invocationId\":" + std::to_string(log_invocation_id) +
                ",\"operationId\":" + std::to_string(log_operation_id) +
                ",\"droppedBefore\":0}\n";
            sink->enqueue(level, std::vector<std::uint8_t>(document.begin(),
                                                           document.end()));
            return;
        }
        const auto category_size = utf8_prefix(category.data, category.size,
                                               WMFS_LOG_MAX_CATEGORY_BYTES);
        const auto available =
            sink->record_bytes > WMFS_LOG_HEADER_SIZE + category_size
                ? sink->record_bytes - WMFS_LOG_HEADER_SIZE - category_size
                : 0;
        const auto message_size = utf8_prefix(
            message.data, message.size,
            std::min<std::size_t>(available, WMFS_LOG_MAX_MESSAGE_BYTES));
        std::vector<std::uint8_t> packet(WMFS_LOG_HEADER_SIZE + category_size +
                                         message_size);
        store64(packet.data(), WMFS_LOG_MAGIC);
        packet[8] = WMFS_LOG_ABI_MAJOR;
        packet[10] = WMFS_LOG_ABI_MINOR;
        store32(packet.data() + 12, WMFS_LOG_HEADER_SIZE);
        store32(packet.data() + 16, packet.size());
        store32(packet.data() + 20,
                category_size != category.size || message_size != message.size
                    ? WMFS_LOG_RECORD_TRUNCATED
                    : 0);
        store32(packet.data() + 24, level);
        std::uint32_t encoded_fields = 0;
        store32(packet.data() + 32, category_size);
        store32(packet.data() + 36, message_size);
        store64(packet.data() + 44, sequence);
        store64(packet.data() + 52, steady_nanoseconds());
        store64(packet.data() + 60, sink->generation);
        store64(packet.data() + 68, log_submission_id);
        store64(packet.data() + 76, log_invocation_id);
        store64(packet.data() + 84, log_operation_id);
        std::memcpy(packet.data() + WMFS_LOG_HEADER_SIZE, category.data,
                    category_size);
        std::memcpy(packet.data() + WMFS_LOG_HEADER_SIZE + category_size,
                    message.data, message_size);
        for (std::uint32_t index = 0;
             fields && index < field_count && index < WMFS_LOG_MAX_FIELDS;
             ++index) {
            const auto &field = fields[index];
            if (!field.name.data || !field.name.size ||
                field.name.size > WMFS_LOG_MAX_NAME_BYTES ||
                field.kind < WMFS_LOG_FIELD_BOOLEAN ||
                field.kind > WMFS_LOG_FIELD_TEXT)
                continue;
            const auto text_size =
                field.kind == WMFS_LOG_FIELD_TEXT && field.text.data
                    ? utf8_prefix(field.text.data, field.text.size,
                                  sink->record_bytes)
                    : 0;
            const auto required =
                WMFS_LOG_FIELD_SIZE + field.name.size + text_size;
            if (required > sink->record_bytes - packet.size()) {
                packet[20] |= WMFS_LOG_RECORD_TRUNCATED;
                break;
            }
            const auto offset = packet.size();
            packet.resize(offset + required);
            packet[offset] = static_cast<std::uint8_t>(field.kind);
            store32(packet.data() + offset + 4, field.name.size);
            store32(packet.data() + offset + 8, text_size);
            store64(packet.data() + offset + 12,
                    field.kind == WMFS_LOG_FIELD_TEXT ? 0 : field.bits);
            std::memcpy(packet.data() + offset + WMFS_LOG_FIELD_SIZE,
                        field.name.data, field.name.size);
            if (text_size)
                std::memcpy(packet.data() + offset + WMFS_LOG_FIELD_SIZE +
                                field.name.size,
                            field.text.data, text_size);
            ++encoded_fields;
        }
        store32(packet.data() + 16, packet.size());
        store32(packet.data() + 28, encoded_fields);
        sink->enqueue(level, std::move(packet));
    } catch (...) {
    }
}

void require(bool condition, const char *message) {
    if (!condition)
        throw std::invalid_argument(message);
}

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

std::string executable_path() {
    std::array<char, 4096> buffer{};
    const auto length =
        ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    return length < 0
               ? std::string("/proc/self/exe")
               : std::string(buffer.data(), static_cast<std::size_t>(length));
}

std::string sha256(std::string_view value) {
    static constexpr std::uint32_t constants[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::array<std::uint32_t, 8> hash{0x6a09e667, 0xbb67ae85, 0x3c6ef372,
                                      0xa54ff53a, 0x510e527f, 0x9b05688c,
                                      0x1f83d9ab, 0x5be0cd19};
    std::vector<std::uint8_t> bytes(value.begin(), value.end());
    const std::uint64_t bit_length = bytes.size() * UINT64_C(8);
    bytes.push_back(0x80);
    while (bytes.size() % 64 != 56)
        bytes.push_back(0);
    for (int shift = 56; shift >= 0; shift -= 8)
        bytes.push_back(static_cast<std::uint8_t>(bit_length >> shift));

    const auto rotate = [](std::uint32_t input, std::uint32_t bits) {
        return (input >> bits) | (input << (32 - bits));
    };
    for (std::size_t offset = 0; offset < bytes.size(); offset += 64) {
        std::array<std::uint32_t, 64> words{};
        for (std::size_t index = 0; index < 16; ++index) {
            const auto at = offset + index * 4;
            words[index] = (std::uint32_t(bytes[at]) << 24) |
                           (std::uint32_t(bytes[at + 1]) << 16) |
                           (std::uint32_t(bytes[at + 2]) << 8) | bytes[at + 3];
        }
        for (std::size_t index = 16; index < words.size(); ++index) {
            const auto first = rotate(words[index - 15], 7) ^
                               rotate(words[index - 15], 18) ^
                               (words[index - 15] >> 3);
            const auto second = rotate(words[index - 2], 17) ^
                                rotate(words[index - 2], 19) ^
                                (words[index - 2] >> 10);
            words[index] =
                words[index - 16] + first + words[index - 7] + second;
        }
        auto a = hash[0];
        auto b = hash[1];
        auto c = hash[2];
        auto d = hash[3];
        auto e = hash[4];
        auto f = hash[5];
        auto g = hash[6];
        auto h = hash[7];
        for (std::size_t index = 0; index < words.size(); ++index) {
            const auto upper = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            const auto choice = (e & f) ^ (~e & g);
            const auto temporary1 =
                h + upper + choice + constants[index] + words[index];
            const auto lower = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto temporary2 = lower + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        const std::uint32_t state[] = {a, b, c, d, e, f, g, h};
        for (std::size_t index = 0; index < hash.size(); ++index)
            hash[index] += state[index];
    }
    static constexpr char hexadecimal[] = "0123456789abcdef";
    std::string result(64, '0');
    for (std::size_t index = 0; index < hash.size(); ++index)
        for (std::size_t byte = 0; byte < 4; ++byte) {
            const auto value_byte = static_cast<std::uint8_t>(
                hash[index] >> static_cast<unsigned>((3 - byte) * 8));
            result[(index * 4 + byte) * 2] = hexadecimal[value_byte >> 4];
            result[(index * 4 + byte) * 2 + 1] = hexadecimal[value_byte & 15];
        }
    return result;
}

int parse_bootstrap(int argc, char **argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "Usage: wmfs-reference-worker --bootstrap-fd FD\n";
        std::exit(0);
    }
    if (argc != 3 || std::string_view(argv[1]) != "--bootstrap-fd")
        throw std::invalid_argument("Expected only --bootstrap-fd FD");
    return std::stoi(argv[2]);
}

std::vector<UniqueFd> received_fds(msghdr &message) {
    std::vector<UniqueFd> result;
    auto *control = CMSG_FIRSTHDR(&message);
    if (control == nullptr)
        return result;
    require(CMSG_NXTHDR(&message, control) == nullptr,
            "Multiple ancillary messages are unsupported");
    require(control->cmsg_level == SOL_SOCKET &&
                control->cmsg_type == SCM_RIGHTS &&
                control->cmsg_len >= CMSG_LEN(0),
            "Invalid ancillary descriptor data");
    const auto size = control->cmsg_len - CMSG_LEN(0);
    require(size % sizeof(int) == 0, "Malformed descriptor array");
    auto *fds = reinterpret_cast<int *>(CMSG_DATA(control));
    for (std::size_t index = 0; index < size / sizeof(int); ++index)
        result.emplace_back(fds[index]);
    return result;
}

std::pair<std::vector<std::uint8_t>, std::vector<UniqueFd>>
receive_packet(int fd, std::size_t max_fds) {
    std::vector<std::uint8_t> packet(WMFS_CONTROL_MAX_PACKET_BYTES);
    std::vector<std::byte> ancillary(CMSG_SPACE(sizeof(int) * max_fds));
    iovec vector{packet.data(), packet.size()};
    msghdr message{};
    message.msg_iov = &vector;
    message.msg_iovlen = 1;
    message.msg_control = ancillary.data();
    message.msg_controllen = ancillary.size();
    ssize_t received;
    do {
        received = ::recvmsg(fd, &message, MSG_CMSG_CLOEXEC);
    } while (received < 0 && errno == EINTR);
    if (received == 0)
        throw std::runtime_error("Control socket closed");
    if (received < 0)
        throw std::runtime_error("Cannot receive control packet");
    auto descriptors = received_fds(message);
    require((message.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) == 0,
            "Control packet was truncated");
    packet.resize(static_cast<std::size_t>(received));
    return {std::move(packet), std::move(descriptors)};
}

void send_packet(int fd, const std::uint8_t *data, std::size_t size) {
    const auto sent = ::send(fd, data, size, MSG_NOSIGNAL);
    require(sent >= 0 && static_cast<std::size_t>(sent) == size,
            "Control packet was not sent atomically");
}

StartupResources accept_startup(int bootstrap_fd) {
    StartupResources result;
    result.bootstrap = UniqueFd(bootstrap_fd);
    auto received = receive_packet(bootstrap_fd, 8);
    std::array<wmfs_control_descriptor_role_record_v1,
               WMFS_CONTROL_MAX_DESCRIPTOR_ROLES>
        roles{};
    wmfs_control_startup_view_v1 request{};
    const wmfs_control_bytes_v1 bytes{received.first.data(),
                                      received.first.size()};
    require(wmfs_control_decode_startup_v1(bytes, WMFS_CONTROL_STARTUP_REQUEST,
                                           &request, roles.data(),
                                           roles.size()) == 0,
            "Invalid STARTUP_REQUEST");
    const std::array<std::uint16_t, 7> base_roles{
        WMFS_CONTROL_DESCRIPTOR_COMMAND_RING,
        WMFS_CONTROL_DESCRIPTOR_COMMAND_DATA_EVENT,
        WMFS_CONTROL_DESCRIPTOR_COMMAND_SPACE_EVENT,
        WMFS_CONTROL_DESCRIPTOR_COMPLETION_RING,
        WMFS_CONTROL_DESCRIPTOR_COMPLETION_DATA_EVENT,
        WMFS_CONTROL_DESCRIPTOR_COMPLETION_SPACE_EVENT,
        WMFS_CONTROL_DESCRIPTOR_FD_CONTROL};
    const bool logging = request.startup.log_mode != WMFS_CONTROL_LOG_DISABLED;
    const std::size_t expected_count = base_roles.size() + (logging ? 1 : 0);
    require(received.second.size() == expected_count,
            "STARTUP_REQUEST descriptor count differs");
    require(request.startup.descriptor_count == expected_count,
            "STARTUP_REQUEST descriptor role count differs");
    for (std::size_t index = 0; index < base_roles.size(); ++index)
        require(roles[index].role == base_roles[index],
                "STARTUP_REQUEST descriptor role order differs");
    if (logging)
        require(roles[7].role == WMFS_CONTROL_DESCRIPTOR_LOG,
                "STARTUP_REQUEST log descriptor role differs");
    require(request.startup.session_generation != 0 &&
                request.startup.metadata_fingerprint ==
                    WMFS_REFERENCE_METADATA_FINGERPRINT &&
                request.startup.capabilities ==
                    WMFS_REFERENCE_STARTUP_CAPABILITIES &&
                request.startup.operation_count ==
                    WMFS_REFERENCE_OPERATION_COUNT &&
                request.startup.protocol_version ==
                    WMFS_REFERENCE_PROTOCOL_VERSION &&
                request.startup.configuration_schema_version ==
                    WMFS_REFERENCE_CONFIGURATION_SCHEMA_VERSION &&
                request.startup.log_mode <= WMFS_CONTROL_LOG_WORKER_FILE &&
                request.startup.status == WMFS_CONTROL_STATUS_OK &&
                std::memcmp(request.startup.interface_fingerprint,
                            interface_fingerprint_sha256, 32) == 0 &&
                std::memcmp(request.startup.configuration_fingerprint,
                            configuration_fingerprint_sha256, 32) == 0,
            "STARTUP_REQUEST identity differs from generated declarations");

    const std::string configuration(
        reinterpret_cast<const char *>(request.config.data),
        request.config.size);
    const auto *api = wmfs_reference_plugin_get_api(WMFS_PLUGIN_ABI_VERSION);
    require(api != nullptr, "Generated plugin API is unavailable");
    const bool has_lifecycle =
        api->struct_size >=
        offsetof(wmfs_plugin_api_v1, shutdown) + sizeof(api->shutdown);
    if (logging) {
        result.log.reset(new StartupResources::LogSink());
        result.log->fd = std::move(received.second[7]);
        result.log->mode = request.startup.log_mode;
        result.log->generation = request.startup.session_generation;
        if (const char *value = std::getenv("WMFS_LOG_LEVEL"))
            result.log->level = static_cast<std::uint32_t>(std::stoul(value));
        if (const char *value = std::getenv("WMFS_LOG_RECORD_BYTES"))
            result.log->record_bytes = std::stoul(value);
        if (const char *value = std::getenv("WMFS_LOG_QUEUE_CAPACITY"))
            result.log->capacity = std::stoul(value);
        result.log->start();
    }
    if (has_lifecycle && (api->features & WMFS_PLUGIN_FEATURE_INITIALIZE)) {
        require(api->initialize != nullptr,
                "Generated plugin initialize callback is missing");
        char error_data[1024]{};
        wmfs_error_buffer_v1 error{sizeof(error), sizeof(error_data),
                                   error_data, 0, 0};
        wmfs_initialize_args_v1 args{};
        args.struct_size = sizeof(args);
        args.features = api->features;
        args.configuration = {configuration.data(), configuration.size()};
        args.logger.struct_size = sizeof(args.logger);
        if (result.log) {
            args.logger.context = result.log.get();
            args.logger.enabled = &log_enabled;
            args.logger.log = &log_write;
        }
        args.error = &error;
        if (api->initialize(&args) != WMFS_STATUS_OK) {
            const std::size_t error_size =
                std::min<std::size_t>(error.size, error.capacity);
            const std::string message =
                error_size ? std::string(error.data, error_size)
                           : "plugin rejected initialization";
            std::vector<std::uint8_t> rejected(WMFS_CONTROL_MAX_PACKET_BYTES);
            wmfs_control_mutable_bytes_v1 output{rejected.data(),
                                                 rejected.size(), 0};
            const wmfs_control_error_v1 control_error{
                WMFS_CONTROL_STATUS_CONFIGURATION_REJECTED,
                static_cast<std::uint32_t>(message.size())};
            const wmfs_control_bytes_v1 message_bytes{
                reinterpret_cast<const std::uint8_t *>(message.data()),
                message.size()};
            require(wmfs_control_encode_error_v1(request.request_id,
                                                 &control_error, message_bytes,
                                                 &output) == 0,
                    "Cannot encode startup rejection");
            send_packet(bootstrap_fd, rejected.data(), output.size);
            throw std::runtime_error("Plugin rejected initialization");
        }
        result.initialized = true;
    }
    const std::string environment =
        std::string("{\"configuration\":") + configuration +
        ",\"configurationDigest\":\"" + sha256(configuration) +
        "\",\"hookAccepted\":true" + ",\"executable\":\"" + executable_path() +
        "\",\"glibcVersion\":\"" + gnu_get_libc_version() +
        "\",\"pythonVersion\":\"none\",\"torchVersion\":\"" +
        WMFS_STRINGIFY(TORCH_VERSION_MAJOR) "." WMFS_STRINGIFY(
            TORCH_VERSION_MINOR) "." WMFS_STRINGIFY(TORCH_VERSION_PATCH) "\"}";
    auto response = request.startup;
    response.descriptor_count = 0;
    response.config_length = environment.size();
    std::vector<std::uint8_t> encoded(WMFS_CONTROL_MAX_PACKET_BYTES);
    wmfs_control_mutable_bytes_v1 output{encoded.data(), encoded.size(), 0};
    const wmfs_control_bytes_v1 environment_bytes{
        reinterpret_cast<const std::uint8_t *>(environment.data()),
        environment.size()};
    require(wmfs_control_encode_startup_v1(
                WMFS_CONTROL_STARTUP_RESPONSE, request.request_id, &response,
                nullptr, environment_bytes, &output) == 0,
            "Cannot encode STARTUP_RESPONSE");
    send_packet(bootstrap_fd, encoded.data(), output.size);

    result.generation = request.startup.session_generation;
    result.api = api;
    for (std::size_t index = 0; index < 3; ++index)
        result.commands[index] = std::move(received.second[index]);
    for (std::size_t index = 0; index < 3; ++index)
        result.completions[index] = std::move(received.second[index + 3]);
    result.fd_control = std::move(received.second[6]);
    return result;
}

std::uint32_t abi_dtype(at::ScalarType dtype) {
    switch (dtype) {
    case at::kFloat:
        return WMFS_DTYPE_FLOAT32;
    case at::kDouble:
        return WMFS_DTYPE_FLOAT64;
    case at::kLong:
        return WMFS_DTYPE_INT64;
    case at::kByte:
        return WMFS_DTYPE_UINT8;
    default:
        throw std::invalid_argument("Unsupported tensor dtype");
    }
}

wmfs_tensor_v1 abi_tensor(at::Tensor &tensor) {
    wmfs_tensor_v1 result{};
    result.struct_size = sizeof(result);
    result.dtype = abi_dtype(tensor.scalar_type());
    result.rank = static_cast<std::uint32_t>(tensor.dim());
    result.byte_length = tensor.nbytes();
    result.data = tensor.data_ptr();
    for (std::uint32_t index = 0; index < result.rank; ++index) {
        result.shape[index] = tensor.size(index);
        result.strides[index] = tensor.stride(index);
    }
    return result;
}

std::vector<wmfs_scalar_v1> scalars(const wmfs_ring_record_v1 &command) {
    std::vector<wmfs_scalar_v1> result;
    for (std::uint16_t index = 0; index < command.scalar_count; ++index) {
        const auto &source = command.scalars[index];
        wmfs_scalar_v1 value{};
        value.struct_size = sizeof(value);
        value.parameter_index = source.parameter_index;
        value.bits = source.bits;
        value.kind =
            source.kind == WMFS_RING_SCALAR_BOOLEAN   ? WMFS_SCALAR_BOOLEAN
            : source.kind == WMFS_RING_SCALAR_FLOAT64 ? WMFS_SCALAR_FLOAT64
                                                      : WMFS_SCALAR_INT64;
        result.push_back(value);
    }
    return result;
}

void dispatch(std::uint32_t operation_id, std::vector<TensorLease> &inputs,
              std::vector<TensorLease> &outputs,
              const std::vector<wmfs_scalar_v1> &scalar_values) {
    std::vector<wmfs_tensor_v1> input_values;
    std::vector<wmfs_tensor_v1> output_values;
    for (auto &input : inputs)
        input_values.push_back(abi_tensor(input.tensor()));
    for (auto &output : outputs)
        output_values.push_back(abi_tensor(output.tensor()));
    wmfs_invocation_v1 invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.operation_id = operation_id;
    invocation.input_count = input_values.size();
    invocation.output_count = output_values.size();
    invocation.scalar_count = scalar_values.size();
    invocation.inputs = input_values.data();
    invocation.outputs = output_values.data();
    invocation.scalars = scalar_values.data();
    const auto *api = wmfs_reference_plugin_get_api(WMFS_PLUGIN_ABI_VERSION);
    require(api != nullptr && api->dispatch(&invocation) == WMFS_STATUS_OK,
            "Generated plugin dispatch rejected invocation");
}

std::vector<wmfs_output_plan_v1>
plan(std::uint32_t operation_id, std::vector<TensorLease> &inputs,
     const std::vector<wmfs_scalar_v1> &scalar_values) {
    std::vector<wmfs_tensor_v1> input_values;
    for (auto &input : inputs)
        input_values.push_back(abi_tensor(input.tensor()));
    wmfs_invocation_v1 invocation{};
    invocation.struct_size = sizeof(invocation);
    invocation.operation_id = operation_id;
    invocation.input_count = input_values.size();
    invocation.scalar_count = scalar_values.size();
    invocation.inputs = input_values.data();
    invocation.scalars = scalar_values.data();
    std::vector<wmfs_output_plan_v1> result(WMFS_PLUGIN_MAX_OUTPUTS);
    std::uint32_t count = 0;
    const auto *api = wmfs_reference_plugin_get_api(WMFS_PLUGIN_ABI_VERSION);
    require(api != nullptr && api->plan_outputs != nullptr &&
                api->plan_outputs(&invocation, result.data(), result.size(),
                                  &count) == WMFS_STATUS_OK &&
                count <= result.size(),
            "Generated output planner rejected invocation");
    result.resize(count);
    return result;
}

TensorLease ring_tensor(MappedBufferCache &buffers,
                        const wmfs_ring_tensor_descriptor_v1 &source,
                        std::uint64_t invocation_id, bool writable) {
    TensorDescriptor descriptor{
        source.buffer_id,
        static_cast<std::uint32_t>(source.buffer_generation),
        source.allocation_id,
        source.byte_offset,
        source.byte_length,
        source.dtype,
        {},
        {}};
    for (std::uint16_t index = 0; index < source.rank; ++index) {
        require(source.shape[index] > 0, "Ring tensor shape is invalid");
        descriptor.shape.push_back(source.shape[index]);
        descriptor.strides.push_back(source.strides[index]);
    }
    return buffers.tensor(descriptor, invocation_id, writable);
}

void set_error(wmfs_ring_record_v1 &completion, std::uint32_t status,
               std::string_view type, std::string_view message) {
    completion.status = status;
    const auto type_size = std::min(type.size(), sizeof(completion.error.type));
    const auto message_size =
        std::min(message.size(), sizeof(completion.error.message));
    completion.error.type_length = type_size;
    completion.error.message_length = message_size;
    std::memcpy(completion.error.type, type.data(), type_size);
    std::memcpy(completion.error.message, message.data(), message_size);
}

template <bool Profiled, bool Logging>
void execute_ring_command(const wmfs_ring_record_v1 &command,
                          wmfs_ring_record_v1 &completion,
                          MappedBufferCache &buffers) {
    if constexpr (Logging) {
        log_submission_id = command.submission_id;
        log_invocation_id = command.invocation_id;
        log_operation_id = command.operation_id;
    }
    c10::InferenceMode inference_mode;
    std::vector<TensorLease> inputs;
    std::vector<TensorLease> outputs;
    std::chrono::steady_clock::time_point started{};
    if constexpr (Profiled)
        started = std::chrono::steady_clock::now();
    for (std::uint16_t index = 0; index < command.tensor_count; ++index) {
        const auto &item = command.tensors[index];
        std::chrono::steady_clock::time_point view_started{};
        if constexpr (Profiled)
            view_started = std::chrono::steady_clock::now();
        if (item.kind == WMFS_RING_TENSOR_INPUT) {
            inputs.push_back(
                ring_tensor(buffers, item, command.invocation_id,
                            item.flags & WMFS_RING_TENSOR_FLAG_WRITABLE));
            if constexpr (Profiled)
                completion.profile.worker_input_views_ns +=
                    nanoseconds_since(view_started);
        } else if (item.kind == WMFS_RING_TENSOR_OUTPUT) {
            outputs.push_back(
                ring_tensor(buffers, item, command.invocation_id, true));
            if constexpr (Profiled)
                completion.profile.worker_output_views_ns +=
                    nanoseconds_since(view_started);
        } else {
            throw std::invalid_argument("Invalid ring tensor kind");
        }
    }
    auto scalar_values = scalars(command);
    if (command.kind == WMFS_RING_COMMAND_PLAN_OUTPUTS) {
        auto planned = plan(command.operation_id, inputs, scalar_values);
        completion.planned_output_count = planned.size();
        for (std::size_t index = 0; index < planned.size(); ++index) {
            completion.planned_outputs[index].dtype = planned[index].dtype;
            completion.planned_outputs[index].rank = planned[index].rank;
            completion.planned_outputs[index].output_index =
                planned[index].output_index;
            for (std::uint32_t axis = 0; axis < planned[index].rank; ++axis)
                completion.planned_outputs[index].shape[axis] =
                    planned[index].shape[axis];
        }
    } else if (command.kind == WMFS_RING_COMMAND_INVOKE) {
        std::chrono::steady_clock::time_point kernel{};
        if constexpr (Profiled)
            kernel = std::chrono::steady_clock::now();
        dispatch(command.operation_id, inputs, outputs, scalar_values);
        if constexpr (Profiled) {
            completion.profile.worker_kernel_ns = nanoseconds_since(kernel);
            const auto elapsed = nanoseconds_since(started);
            completion.profile.worker_dispatch_ns =
                elapsed > completion.profile.worker_kernel_ns
                    ? elapsed - completion.profile.worker_kernel_ns
                    : 0;
        }
    } else if (command.kind == WMFS_RING_COMMAND_PING) {
        std::chrono::steady_clock::time_point kernel{};
        if constexpr (Profiled)
            kernel = std::chrono::steady_clock::now();
        if (command.operation_id)
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(command.operation_id));
        if constexpr (Profiled)
            completion.profile.worker_kernel_ns = nanoseconds_since(kernel);
    } else {
        throw std::invalid_argument("Unsupported ring command");
    }
}

void run_ring(RingConsumer commands, RingProducer completions,
              MappedBufferCache &buffers, bool logging_enabled) {
    for (;;) {
        wmfs_ring_record_v1 command{};
        if (commands.pop(command) != RingWaitResult::success)
            return;
        const bool profiled = command.flags & WMFS_RING_RECORD_FLAG_PROFILE;
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
        completion.profile.worker_dequeued_ns =
            profiled ? steady_nanoseconds() : 0;
        completion.profile.worker_started_ns =
            profiled ? steady_nanoseconds() : 0;
        try {
            if (profiled && logging_enabled)
                execute_ring_command<true, true>(command, completion, buffers);
            else if (profiled)
                execute_ring_command<true, false>(command, completion, buffers);
            else if (logging_enabled)
                execute_ring_command<false, true>(command, completion, buffers);
            else
                execute_ring_command<false, false>(command, completion,
                                                   buffers);
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
        if (command.kind == WMFS_RING_COMMAND_INVOKE)
            buffers.finish_invocation(command.invocation_id);
        completion.profile.completion_published_ns =
            profiled ? steady_nanoseconds() : 0;
        if (completions.push(completion) != RingWaitResult::success)
            return;
    }
}

std::uint64_t serve_lifecycle(int fd) {
    for (;;) {
        auto received = receive_packet(fd, 1);
        require(received.second.empty(),
                "Lifecycle frame carried file descriptors");
        const wmfs_control_bytes_v1 packet{received.first.data(),
                                           received.first.size()};
        std::uint64_t request_id = 0;
        if (wmfs_control_decode_empty_v1(packet, WMFS_CONTROL_PING,
                                         &request_id) == 0) {
            std::array<std::uint8_t, WMFS_CONTROL_FRAME_HEADER_SIZE> response{};
            wmfs_control_mutable_bytes_v1 output{response.data(),
                                                 response.size(), 0};
            require(wmfs_control_encode_empty_v1(WMFS_CONTROL_PONG, request_id,
                                                 &output) == 0,
                    "Cannot encode lifecycle response");
            send_packet(fd, response.data(), output.size);
        } else if (wmfs_control_decode_empty_v1(packet, WMFS_CONTROL_SHUTDOWN,
                                                &request_id) == 0) {
            return request_id;
        } else {
            throw std::runtime_error("Invalid lifecycle frame");
        }
    }
}

} // namespace

int run_worker(int argc, char **argv) {
    auto resources = accept_startup(parse_bootstrap(argc, argv));
    auto commands = RingConsumer::borrow(
        std::move(resources.commands[0]), std::move(resources.commands[1]),
        std::move(resources.commands[2]), resources.generation);
    auto command_interrupt = RingProducer::borrow(
        commands.duplicate_ring_fd(), commands.duplicate_data_event_fd(),
        commands.duplicate_space_event_fd(), resources.generation);
    auto completions = RingProducer::borrow(std::move(resources.completions[0]),
                                            std::move(resources.completions[1]),
                                            std::move(resources.completions[2]),
                                            resources.generation);
    MappedBufferCache buffers;
    std::exception_ptr receiver_error;
    std::thread receiver([&] {
        try {
            receive_buffer_transfers(resources.fd_control.get(),
                                     resources.generation, buffers);
        } catch (...) {
            receiver_error = std::current_exception();
            ::shutdown(resources.bootstrap.get(), SHUT_RDWR);
        }
    });
    std::thread ring_worker([&] {
        try {
            run_ring(std::move(commands), std::move(completions), buffers,
                     resources.log != nullptr);
        } catch (...) {
            ::shutdown(resources.bootstrap.get(), SHUT_RDWR);
        }
    });
    try {
        const auto shutdown_request =
            serve_lifecycle(resources.bootstrap.get());
        ::shutdown(resources.fd_control.get(), SHUT_RDWR);
        command_interrupt.close();
        receiver.join();
        ring_worker.join();
        if (resources.initialized && resources.api && resources.api->shutdown)
            resources.api->shutdown();
        std::array<std::uint8_t, WMFS_CONTROL_FRAME_HEADER_SIZE> response{};
        wmfs_control_mutable_bytes_v1 output{response.data(), response.size(),
                                             0};
        require(wmfs_control_encode_empty_v1(WMFS_CONTROL_SHUTDOWN_ACK,
                                             shutdown_request, &output) == 0,
                "Cannot encode shutdown acknowledgement");
        send_packet(resources.bootstrap.get(), response.data(), output.size);
    } catch (...) {
        ::shutdown(resources.fd_control.get(), SHUT_RDWR);
        command_interrupt.close();
        if (receiver.joinable())
            receiver.join();
        if (ring_worker.joinable())
            ring_worker.join();
        throw;
    }
    if (receiver_error)
        std::rethrow_exception(receiver_error);
    return 0;
}

} // namespace wmfs::reference

int main(int argc, char **argv) {
    try {
        return wmfs::reference::run_worker(argc, argv);
    } catch (const std::exception &error) {
        std::cerr << "wmfs-reference-worker: " << error.what() << '\n';
        return 1;
    }
}
