#include <wmfs/protocol/control.h>

#include <cstring>
#include <limits>

namespace {
enum { OK = 0, INVALID = 1, NO_SPACE = 2 };

uint16_t load16(const uint8_t *p) {
    return static_cast<uint16_t>(p[0]) |
           static_cast<uint16_t>(static_cast<uint16_t>(p[1]) << 8);
}
uint32_t load32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t load64(const uint8_t *p) {
    return static_cast<uint64_t>(load32(p)) |
           (static_cast<uint64_t>(load32(p + 4)) << 32);
}
void store16(uint8_t *p, uint16_t value) {
    p[0] = static_cast<uint8_t>(value);
    p[1] = static_cast<uint8_t>(value >> 8);
}
void store32(uint8_t *p, uint32_t value) {
    p[0] = static_cast<uint8_t>(value);
    p[1] = static_cast<uint8_t>(value >> 8);
    p[2] = static_cast<uint8_t>(value >> 16);
    p[3] = static_cast<uint8_t>(value >> 24);
}
void store64(uint8_t *p, uint64_t value) {
    store32(p, static_cast<uint32_t>(value));
    store32(p + 4, static_cast<uint32_t>(value >> 32));
}
bool valid_utf8(const uint8_t *p, size_t size) {
    size_t i = 0;
    while (i < size) {
        const uint8_t first = p[i++];
        if (first < 0x80)
            continue;
        uint32_t value = 0;
        size_t extra = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            value = first & 0x1f;
            extra = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            value = first & 0x0f;
            extra = 2;
        } else if (first >= 0xf0 && first <= 0xf4) {
            value = first & 0x07;
            extra = 3;
        } else
            return false;
        if (extra > size - i)
            return false;
        for (size_t j = 0; j < extra; ++j) {
            const uint8_t next = p[i++];
            if ((next & 0xc0) != 0x80)
                return false;
            value = (value << 6) | (next & 0x3f);
        }
        if ((extra == 2 && value < 0x800) || (extra == 3 && value < 0x10000) ||
            value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
            return false;
    }
    return true;
}
bool frame(const wmfs_control_bytes_v1 &packet, uint16_t kind,
           const uint8_t **payload, uint32_t *payload_size,
           uint64_t *request_id) {
    if (!packet.data || packet.size < 32 ||
        packet.size > WMFS_CONTROL_MAX_PACKET_BYTES)
        return false;
    const uint8_t *p = packet.data;
    if (load64(p) != WMFS_CONTROL_MAGIC ||
        load16(p + 8) != WMFS_CONTROL_ABI_MAJOR ||
        load16(p + 10) > WMFS_CONTROL_ABI_MINOR || load16(p + 12) != kind ||
        load16(p + 14) != 0 || load32(p + 16) != 32)
        return false;
    const uint32_t length = load32(p + 20);
    if (static_cast<size_t>(length) != packet.size - 32)
        return false;
    *payload = p + 32;
    *payload_size = length;
    if (request_id != 0)
        *request_id = load64(p + 24);
    return true;
}
bool begin(uint16_t kind, uint64_t request_id, uint32_t payload_size,
           wmfs_control_mutable_bytes_v1 *out, uint8_t **payload) {
    if (!out || !out->data ||
        payload_size > WMFS_CONTROL_MAX_PACKET_BYTES - 32 ||
        out->capacity < static_cast<size_t>(32) + payload_size)
        return false;
    uint8_t *p = out->data;
    store64(p, WMFS_CONTROL_MAGIC);
    store16(p + 8, WMFS_CONTROL_ABI_MAJOR);
    store16(p + 10, WMFS_CONTROL_ABI_MINOR);
    store16(p + 12, kind);
    store16(p + 14, 0);
    store32(p + 16, 32);
    store32(p + 20, payload_size);
    store64(p + 24, request_id);
    out->size = 32 + payload_size;
    *payload = p + 32;
    return true;
}
void startup_store(uint8_t *p, const wmfs_control_startup_v1 &s) {
    store64(p, s.session_generation);
    std::memcpy(p + 8, s.interface_fingerprint, 32);
    std::memcpy(p + 40, s.configuration_fingerprint, 32);
    store64(p + 72, s.metadata_fingerprint);
    store64(p + 80, s.capabilities);
    store32(p + 88, s.operation_count);
    store32(p + 92, s.protocol_version);
    store32(p + 96, s.configuration_schema_version);
    store32(p + 100, s.config_length);
    store16(p + 104, s.descriptor_count);
    store16(p + 106, s.log_mode);
    store32(p + 108, s.status);
}
void startup_load(const uint8_t *p, wmfs_control_startup_v1 &s) {
    s.session_generation = load64(p);
    std::memcpy(s.interface_fingerprint, p + 8, 32);
    std::memcpy(s.configuration_fingerprint, p + 40, 32);
    s.metadata_fingerprint = load64(p + 72);
    s.capabilities = load64(p + 80);
    s.operation_count = load32(p + 88);
    s.protocol_version = load32(p + 92);
    s.configuration_schema_version = load32(p + 96);
    s.config_length = load32(p + 100);
    s.descriptor_count = load16(p + 104);
    s.log_mode = load16(p + 106);
    s.status = load32(p + 108);
}
} // namespace

extern "C" int32_t wmfs_control_encode_startup_v1(
    uint16_t kind, uint64_t request_id, const wmfs_control_startup_v1 *s,
    const wmfs_control_descriptor_role_record_v1 *roles,
    wmfs_control_bytes_v1 config, wmfs_control_mutable_bytes_v1 *out) {
    if (!s || s->config_length != config.size ||
        config.size > WMFS_CONTROL_MAX_CONFIG_BYTES ||
        s->descriptor_count > WMFS_CONTROL_MAX_DESCRIPTOR_ROLES ||
        (s->descriptor_count && !roles) || (config.size && !config.data) ||
        (kind != WMFS_CONTROL_STARTUP_REQUEST &&
         kind != WMFS_CONTROL_STARTUP_RESPONSE) ||
        s->log_mode > WMFS_CONTROL_LOG_WORKER_FILE ||
        !valid_utf8(config.data, config.size))
        return INVALID;
    const size_t payload_size =
        112 + static_cast<size_t>(s->descriptor_count) * 8 + config.size;
    if (payload_size > std::numeric_limits<uint32_t>::max())
        return INVALID;
    uint8_t *p = 0;
    if (!begin(kind, request_id, static_cast<uint32_t>(payload_size), out, &p))
        return NO_SPACE;
    startup_store(p, *s);
    p += 112;
    for (uint16_t i = 0; i < s->descriptor_count; ++i, p += 8) {
        if (roles[i].role == 0 || roles[i].role > WMFS_CONTROL_DESCRIPTOR_LOG ||
            roles[i].flags || roles[i].reserved)
            return INVALID;
        store16(p, roles[i].role);
        store16(p + 2, 0);
        store32(p + 4, 0);
    }
    if (config.size)
        std::memcpy(p, config.data, config.size);
    return OK;
}

extern "C" int32_t
wmfs_control_decode_startup_v1(wmfs_control_bytes_v1 packet, uint16_t kind,
                               wmfs_control_startup_view_v1 *out,
                               wmfs_control_descriptor_role_record_v1 *roles,
                               size_t role_capacity) {
    const uint8_t *p = 0;
    uint32_t size = 0;
    if (!out || !frame(packet, kind, &p, &size, &out->request_id) || size < 112)
        return INVALID;
    startup_load(p, out->startup);
    const wmfs_control_startup_v1 &s = out->startup;
    if (s.config_length > WMFS_CONTROL_MAX_CONFIG_BYTES ||
        s.descriptor_count > WMFS_CONTROL_MAX_DESCRIPTOR_ROLES ||
        s.descriptor_count > role_capacity || (s.descriptor_count && !roles) ||
        s.log_mode > WMFS_CONTROL_LOG_WORKER_FILE ||
        static_cast<size_t>(size) !=
            112 + static_cast<size_t>(s.descriptor_count) * 8 + s.config_length)
        return INVALID;
    p += 112;
    for (uint16_t i = 0; i < s.descriptor_count; ++i, p += 8) {
        roles[i].role = load16(p);
        roles[i].flags = load16(p + 2);
        roles[i].reserved = load32(p + 4);
        if (!roles[i].role || roles[i].role > WMFS_CONTROL_DESCRIPTOR_LOG ||
            roles[i].flags || roles[i].reserved)
            return INVALID;
    }
    if (!valid_utf8(p, s.config_length))
        return INVALID;
    out->roles = roles;
    out->config.data = p;
    out->config.size = s.config_length;
    return OK;
}

extern "C" int32_t
wmfs_control_encode_fd_batch_v1(uint64_t request_id,
                                const wmfs_control_fd_batch_v1 *batch,
                                const wmfs_control_fd_entry_v1 *entries,
                                wmfs_control_mutable_bytes_v1 *out) {
    if (!batch || !batch->entry_count ||
        batch->entry_count > WMFS_CONTROL_MAX_FD_ENTRIES || !entries ||
        batch->reserved ||
        batch->flags != WMFS_CONTROL_FD_BATCH_FLAG_TRANSACTIONAL)
        return INVALID;
    uint16_t maps = 0;
    for (uint16_t i = 0; i < batch->entry_count; ++i) {
        if (entries[i].kind == WMFS_CONTROL_FD_ENTRY_MAP)
            ++maps;
        else if (entries[i].kind != WMFS_CONTROL_FD_ENTRY_RETIRE)
            return INVALID;
        if (entries[i].flags &
            ~(WMFS_CONTROL_FD_FLAG_WRITABLE | WMFS_CONTROL_FD_FLAG_ARENA))
            return INVALID;
        if (entries[i].kind == WMFS_CONTROL_FD_ENTRY_RETIRE && entries[i].flags)
            return INVALID;
    }
    if (maps != batch->fd_count)
        return INVALID;
    const uint32_t size = 32 + static_cast<uint32_t>(batch->entry_count) * 48;
    uint8_t *p = 0;
    if (!begin(WMFS_CONTROL_FD_TRANSFER, request_id, size, out, &p))
        return NO_SPACE;
    store64(p, batch->transfer_id);
    store64(p + 8, batch->session_generation);
    store16(p + 16, batch->entry_count);
    store16(p + 18, batch->fd_count);
    store32(p + 20, batch->flags);
    store64(p + 24, 0);
    p += 32;
    for (uint16_t i = 0; i < batch->entry_count; ++i, p += 48) {
        store32(p, entries[i].kind);
        store32(p + 4, entries[i].flags);
        store64(p + 8, entries[i].buffer_id);
        store64(p + 16, entries[i].generation);
        store64(p + 24, entries[i].allocation_id);
        store64(p + 32, entries[i].invocation_id);
        store64(p + 40, entries[i].byte_length);
    }
    return OK;
}

extern "C" int32_t wmfs_control_decode_fd_batch_v1(
    wmfs_control_bytes_v1 packet, wmfs_control_fd_batch_view_v1 *out,
    wmfs_control_fd_entry_v1 *entries, size_t capacity) {
    const uint8_t *p = 0;
    uint32_t size = 0;
    if (!out ||
        !frame(packet, WMFS_CONTROL_FD_TRANSFER, &p, &size, &out->request_id) ||
        size < 32)
        return INVALID;
    wmfs_control_fd_batch_v1 &b = out->batch;
    b.transfer_id = load64(p);
    b.session_generation = load64(p + 8);
    b.entry_count = load16(p + 16);
    b.fd_count = load16(p + 18);
    b.flags = load32(p + 20);
    b.reserved = load64(p + 24);
    if (!b.entry_count || b.entry_count > WMFS_CONTROL_MAX_FD_ENTRIES ||
        b.entry_count > capacity || !entries || b.reserved ||
        b.flags != WMFS_CONTROL_FD_BATCH_FLAG_TRANSACTIONAL ||
        size != 32 + static_cast<uint32_t>(b.entry_count) * 48)
        return INVALID;
    p += 32;
    uint16_t maps = 0;
    for (uint16_t i = 0; i < b.entry_count; ++i, p += 48) {
        wmfs_control_fd_entry_v1 &e = entries[i];
        e.kind = load32(p);
        e.flags = load32(p + 4);
        e.buffer_id = load64(p + 8);
        e.generation = load64(p + 16);
        e.allocation_id = load64(p + 24);
        e.invocation_id = load64(p + 32);
        e.byte_length = load64(p + 40);
        if (e.kind == WMFS_CONTROL_FD_ENTRY_MAP)
            ++maps;
        else if (e.kind != WMFS_CONTROL_FD_ENTRY_RETIRE)
            return INVALID;
        if (e.flags &
                ~(WMFS_CONTROL_FD_FLAG_WRITABLE | WMFS_CONTROL_FD_FLAG_ARENA) ||
            (e.kind == WMFS_CONTROL_FD_ENTRY_RETIRE && e.flags))
            return INVALID;
    }
    if (maps != b.fd_count)
        return INVALID;
    out->entries = entries;
    return OK;
}

extern "C" int32_t wmfs_control_encode_fd_ack_v1(
    uint64_t request_id, const wmfs_control_fd_ack_v1 *ack,
    wmfs_control_bytes_v1 error, wmfs_control_mutable_bytes_v1 *out) {
    if (!ack || ack->error_length != error.size ||
        error.size > WMFS_CONTROL_MAX_ERROR_BYTES ||
        (error.size && !error.data) || ack->reserved || ack->flags ||
        !valid_utf8(error.data, error.size))
        return INVALID;
    uint8_t *p = 0;
    if (!begin(WMFS_CONTROL_FD_TRANSFER_ACK, request_id,
               32 + static_cast<uint32_t>(error.size), out, &p))
        return NO_SPACE;
    store64(p, ack->transfer_id);
    store64(p + 8, ack->session_generation);
    store32(p + 16, ack->status);
    store32(p + 20, 0);
    store32(p + 24, ack->error_length);
    store32(p + 28, 0);
    if (error.size)
        std::memcpy(p + 32, error.data, error.size);
    return OK;
}

extern "C" int32_t wmfs_control_decode_fd_ack_v1(wmfs_control_bytes_v1 packet,
                                                 uint64_t *request_id,
                                                 wmfs_control_fd_ack_v1 *ack,
                                                 wmfs_control_bytes_v1 *error) {
    const uint8_t *p = 0;
    uint32_t size = 0;
    if (!request_id || !ack || !error ||
        !frame(packet, WMFS_CONTROL_FD_TRANSFER_ACK, &p, &size, request_id) ||
        size < 32)
        return INVALID;
    ack->transfer_id = load64(p);
    ack->session_generation = load64(p + 8);
    ack->status = load32(p + 16);
    ack->flags = load32(p + 20);
    ack->error_length = load32(p + 24);
    ack->reserved = load32(p + 28);
    if (ack->flags || ack->reserved ||
        ack->error_length > WMFS_CONTROL_MAX_ERROR_BYTES ||
        size != 32 + ack->error_length ||
        !valid_utf8(p + 32, ack->error_length))
        return INVALID;
    error->data = p + 32;
    error->size = ack->error_length;
    return OK;
}

extern "C" int32_t wmfs_control_encode_error_v1(
    uint64_t request_id, const wmfs_control_error_v1 *error,
    wmfs_control_bytes_v1 message, wmfs_control_mutable_bytes_v1 *out) {
    if (!error || error->status == WMFS_CONTROL_STATUS_OK ||
        error->error_length != message.size ||
        message.size > WMFS_CONTROL_MAX_ERROR_BYTES ||
        (message.size && !message.data) ||
        !valid_utf8(message.data, message.size))
        return INVALID;
    uint8_t *p = 0;
    if (!begin(WMFS_CONTROL_ERROR_RESPONSE, request_id,
               8 + static_cast<uint32_t>(message.size), out, &p))
        return NO_SPACE;
    store32(p, error->status);
    store32(p + 4, error->error_length);
    if (message.size)
        std::memcpy(p + 8, message.data, message.size);
    return OK;
}

extern "C" int32_t
wmfs_control_decode_error_v1(wmfs_control_bytes_v1 packet, uint64_t *request_id,
                             wmfs_control_error_v1 *error,
                             wmfs_control_bytes_v1 *message) {
    const uint8_t *p = 0;
    uint32_t size = 0;
    if (!request_id || !error || !message ||
        !frame(packet, WMFS_CONTROL_ERROR_RESPONSE, &p, &size, request_id) ||
        size < 8)
        return INVALID;
    error->status = load32(p);
    error->error_length = load32(p + 4);
    if (error->status == WMFS_CONTROL_STATUS_OK ||
        error->error_length > WMFS_CONTROL_MAX_ERROR_BYTES ||
        size != 8 + error->error_length ||
        !valid_utf8(p + 8, error->error_length))
        return INVALID;
    message->data = p + 8;
    message->size = error->error_length;
    return OK;
}

extern "C" int32_t
wmfs_control_encode_empty_v1(uint16_t kind, uint64_t request_id,
                             wmfs_control_mutable_bytes_v1 *out) {
    if (kind != WMFS_CONTROL_PING && kind != WMFS_CONTROL_PONG &&
        kind != WMFS_CONTROL_SHUTDOWN && kind != WMFS_CONTROL_SHUTDOWN_ACK)
        return INVALID;
    uint8_t *payload = 0;
    return begin(kind, request_id, 0, out, &payload) ? OK : NO_SPACE;
}

extern "C" int32_t wmfs_control_decode_empty_v1(wmfs_control_bytes_v1 packet,
                                                uint16_t kind,
                                                uint64_t *request_id) {
    const uint8_t *payload = 0;
    uint32_t payload_size = 0;
    if (!request_id ||
        (kind != WMFS_CONTROL_PING && kind != WMFS_CONTROL_PONG &&
         kind != WMFS_CONTROL_SHUTDOWN && kind != WMFS_CONTROL_SHUTDOWN_ACK) ||
        !frame(packet, kind, &payload, &payload_size, request_id) ||
        payload_size)
        return INVALID;
    return OK;
}
