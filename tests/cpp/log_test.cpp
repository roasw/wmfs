#include <wmfs/protocol/log.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
void store32(std::uint8_t *p, std::uint32_t value) {
    for (unsigned index = 0; index < 4; ++index)
        p[index] = static_cast<std::uint8_t>(value >> (index * 8));
}
void store64(std::uint8_t *p, std::uint64_t value) {
    for (unsigned index = 0; index < 8; ++index)
        p[index] = static_cast<std::uint8_t>(value >> (index * 8));
}
} // namespace

int main() {
    const char category[] = "codec";
    const char message[] = "parity";
    const char name[] = "signed";
    std::vector<std::uint8_t> packet(WMFS_LOG_HEADER_SIZE + sizeof(category) -
                                     1 + sizeof(message) - 1 +
                                     WMFS_LOG_FIELD_SIZE + sizeof(name) - 1);
    store64(packet.data(), WMFS_LOG_MAGIC);
    packet[8] = WMFS_LOG_ABI_MAJOR;
    packet[10] = WMFS_LOG_ABI_MINOR;
    store32(packet.data() + 12, WMFS_LOG_HEADER_SIZE);
    store32(packet.data() + 16, packet.size());
    store32(packet.data() + 24, 30);
    store32(packet.data() + 28, 1);
    store32(packet.data() + 32, sizeof(category) - 1);
    store32(packet.data() + 36, sizeof(message) - 1);
    store64(packet.data() + 44, 4);
    store64(packet.data() + 52, 5);
    store64(packet.data() + 60, 6);
    store64(packet.data() + 68, 7);
    store64(packet.data() + 76, 8);
    store64(packet.data() + 84, 9);
    auto offset = std::size_t(WMFS_LOG_HEADER_SIZE);
    std::memcpy(packet.data() + offset, category, sizeof(category) - 1);
    offset += sizeof(category) - 1;
    std::memcpy(packet.data() + offset, message, sizeof(message) - 1);
    offset += sizeof(message) - 1;
    packet[offset] = WMFS_LOG_WIRE_INT64;
    store32(packet.data() + offset + 4, sizeof(name) - 1);
    store64(packet.data() + offset + 12, UINT64_MAX - 1);
    std::memcpy(packet.data() + offset + WMFS_LOG_FIELD_SIZE, name,
                sizeof(name) - 1);

    wmfs_log_record_v1 record{};
    std::array<wmfs_log_field_view_v1, 1> fields{};
    const wmfs_log_bytes_v1 bytes{packet.data(), packet.size()};
    assert(wmfs_log_decode_v1(bytes, &record, fields.data(), fields.size()) ==
           0);
    assert(record.level == 30 && record.session_id == 6 &&
           record.submission_id == 7 && record.invocation_id == 8 &&
           record.operation_id == 9);
    assert(fields[0].kind == WMFS_LOG_WIRE_INT64 &&
           static_cast<std::int64_t>(fields[0].bits) == -2);

    auto malformed = packet;
    malformed.push_back(0);
    assert(wmfs_log_decode_v1({malformed.data(), malformed.size()}, &record,
                              fields.data(), fields.size()) == 1);
    malformed = packet;
    store32(malformed.data() + 20, UINT32_C(0x80000000));
    assert(wmfs_log_decode_v1({malformed.data(), malformed.size()}, &record,
                              fields.data(), fields.size()) == 1);
    malformed = packet;
    malformed[WMFS_LOG_HEADER_SIZE] = 0xff;
    assert(wmfs_log_decode_v1({malformed.data(), malformed.size()}, &record,
                              fields.data(), fields.size()) == 1);
    assert(wmfs_log_decode_v1(bytes, &record, nullptr, 0) == 1);
}
