#include <wmfs/protocol/control.h>

#include <cassert>
#include <cstring>
#include <vector>

int main() {
    std::vector<uint8_t> empty_bytes(32);
    wmfs_control_mutable_bytes_v1 empty_output = {&empty_bytes[0],
                                                  empty_bytes.size(), 0};
    assert(wmfs_control_encode_empty_v1(WMFS_CONTROL_PING,
                                        UINT64_C(0x0102030405060708),
                                        &empty_output) == 0);
    const uint8_t ping_golden[32] = {
        0x57, 0x4d, 0x46, 0x53, 0x43, 0x54, 0x4c, 0x31, 0x01, 0x00, 0x00,
        0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01};
    assert(std::memcmp(&empty_bytes[0], ping_golden, sizeof(ping_golden)) == 0);
    uint64_t ping_request_id = 0;
    const wmfs_control_bytes_v1 ping_packet = {&empty_bytes[0],
                                               empty_output.size};
    assert(wmfs_control_decode_empty_v1(ping_packet, WMFS_CONTROL_PING,
                                        &ping_request_id) == 0);
    assert(ping_request_id == UINT64_C(0x0102030405060708));

    wmfs_control_startup_v1 startup = {};
    startup.session_generation = UINT64_C(0x0102030405060708);
    startup.metadata_fingerprint = UINT64_C(0x8877665544332211);
    startup.capabilities = WMFS_CONTROL_CAPABILITY_FD_CONTROL;
    startup.operation_count = 6;
    startup.protocol_version = 11;
    startup.configuration_schema_version = 1;
    startup.config_length = 2;
    startup.descriptor_count = 1;
    startup.log_mode = WMFS_CONTROL_LOG_DISABLED;
    for (unsigned i = 0; i < 32; ++i) {
        startup.interface_fingerprint[i] = static_cast<uint8_t>(i);
        startup.configuration_fingerprint[i] = static_cast<uint8_t>(31 - i);
    }
    wmfs_control_descriptor_role_record_v1 role = {};
    role.role = WMFS_CONTROL_DESCRIPTOR_FD_CONTROL;
    const uint8_t config[] = {'{', '}'};
    std::vector<uint8_t> bytes(WMFS_CONTROL_MAX_PACKET_BYTES);
    wmfs_control_mutable_bytes_v1 output = {&bytes[0], bytes.size(), 0};
    const wmfs_control_bytes_v1 config_view = {config, sizeof(config)};
    assert(wmfs_control_encode_startup_v1(WMFS_CONTROL_STARTUP_REQUEST, 9,
                                          &startup, &role, config_view,
                                          &output) == 0);
    /* Golden prefix proves explicit little-endian encoding. */
    const uint8_t prefix[] = {'W', 'M', 'F', 'S', 'C', 'T', 'L', '1', 1, 0};
    assert(std::memcmp(&bytes[0], prefix, sizeof(prefix)) == 0);
    wmfs_control_startup_view_v1 decoded = {};
    wmfs_control_descriptor_role_record_v1 decoded_role = {};
    const wmfs_control_bytes_v1 packet = {&bytes[0], output.size};
    assert(wmfs_control_decode_startup_v1(packet, WMFS_CONTROL_STARTUP_REQUEST,
                                          &decoded, &decoded_role, 1) == 0);
    assert(decoded.startup.session_generation == startup.session_generation);
    assert(decoded.request_id == 9);
    assert(decoded.config.size == 2 && decoded_role.role == role.role);

    startup.config_length = WMFS_CONTROL_MAX_CONFIG_BYTES + 1U;
    assert(wmfs_control_encode_startup_v1(WMFS_CONTROL_STARTUP_REQUEST, 0,
                                          &startup, &role, config_view,
                                          &output) != 0);

    wmfs_control_fd_entry_v1 entries[2] = {};
    entries[0].kind = WMFS_CONTROL_FD_ENTRY_MAP;
    entries[0].generation = UINT64_C(0xffffffffffffffff);
    entries[0].allocation_id = UINT64_C(0x8000000000000000);
    entries[0].invocation_id = UINT64_C(0x0102030405060708);
    entries[0].byte_length = 4096;
    entries[1].kind = WMFS_CONTROL_FD_ENTRY_RETIRE;
    wmfs_control_fd_batch_v1 batch = {};
    batch.transfer_id = 7;
    batch.session_generation = 8;
    batch.entry_count = 2;
    batch.fd_count = 1;
    batch.flags = WMFS_CONTROL_FD_BATCH_FLAG_TRANSACTIONAL;
    assert(wmfs_control_encode_fd_batch_v1(10, &batch, entries, &output) == 0);
    wmfs_control_fd_entry_v1 decoded_entries[2] = {};
    wmfs_control_fd_batch_view_v1 decoded_batch = {};
    const wmfs_control_bytes_v1 fd_packet = {&bytes[0], output.size};
    assert(wmfs_control_decode_fd_batch_v1(fd_packet, &decoded_batch,
                                           decoded_entries, 2) == 0);
    assert(decoded_entries[0].invocation_id == entries[0].invocation_id);
    bytes[20] ^= 1;
    assert(wmfs_control_decode_fd_batch_v1(fd_packet, &decoded_batch,
                                           decoded_entries, 2) != 0);
    return 0;
}
