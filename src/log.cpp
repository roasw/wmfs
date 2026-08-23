#include <wmfs/protocol/log.h>

#include <cstring>

namespace {
uint16_t load16(const uint8_t *p) {
    return uint16_t(p[0]) | uint16_t(uint16_t(p[1]) << 8);
}
uint32_t load32(const uint8_t *p) {
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 |
           uint32_t(p[3]) << 24;
}
uint64_t load64(const uint8_t *p) {
    return uint64_t(load32(p)) | uint64_t(load32(p + 4)) << 32;
}
bool utf8(const uint8_t *p, size_t size) {
    size_t i = 0;
    while (i < size) {
        const uint8_t first = p[i++];
        if (first < 0x80)
            continue;
        uint32_t value = 0;
        size_t extra = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            value = first & 31;
            extra = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            value = first & 15;
            extra = 2;
        } else if (first >= 0xf0 && first <= 0xf4) {
            value = first & 7;
            extra = 3;
        } else
            return false;
        if (extra > size - i)
            return false;
        for (size_t j = 0; j < extra; ++j) {
            const uint8_t next = p[i++];
            if ((next & 0xc0) != 0x80)
                return false;
            value = value << 6 | (next & 63);
        }
        if ((extra == 2 && value < 0x800) || (extra == 3 && value < 0x10000) ||
            value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
            return false;
    }
    return true;
}
} // namespace

int wmfs_log_decode_v1(wmfs_log_bytes_v1 packet, wmfs_log_record_v1 *record,
                       wmfs_log_field_view_v1 *fields, size_t capacity) {
    if (!packet.data || !record || packet.size < WMFS_LOG_HEADER_SIZE ||
        packet.size > WMFS_LOG_MAX_RECORD_BYTES)
        return 1;
    const uint8_t *p = packet.data;
    if (load64(p) != WMFS_LOG_MAGIC || load16(p + 8) != WMFS_LOG_ABI_MAJOR ||
        load16(p + 10) > WMFS_LOG_ABI_MINOR ||
        load32(p + 12) != WMFS_LOG_HEADER_SIZE ||
        load32(p + 16) != packet.size || load32(p + 20) & ~UINT32_C(3))
        return 1;
    const uint32_t level = load32(p + 24), count = load32(p + 28);
    const uint32_t category_size = load32(p + 32),
                   message_size = load32(p + 36);
    if ((level < 10 || level > 50 || level % 10) ||
        count > WMFS_LOG_MAX_FIELDS || count > capacity || (count && !fields) ||
        category_size > WMFS_LOG_MAX_CATEGORY_BYTES ||
        message_size > WMFS_LOG_MAX_MESSAGE_BYTES || load32(p + 40))
        return 1;
    size_t offset = WMFS_LOG_HEADER_SIZE;
    if (category_size + message_size > packet.size - offset ||
        !utf8(p + offset, category_size + message_size))
        return 1;
    std::memset(record, 0, sizeof(*record));
    record->flags = load32(p + 20);
    record->level = level;
    record->field_count = count;
    record->sequence = load64(p + 44);
    record->time_ns = load64(p + 52);
    record->session_id = load64(p + 60);
    record->submission_id = load64(p + 68);
    record->invocation_id = load64(p + 76);
    record->operation_id = load64(p + 84);
    record->dropped_before = load64(p + 92);
    record->category = {p + offset, category_size};
    offset += category_size;
    record->message = {p + offset, message_size};
    offset += message_size;
    for (uint32_t i = 0; i < count; ++i) {
        if (packet.size - offset < WMFS_LOG_FIELD_SIZE)
            return 1;
        const uint16_t kind = load16(p + offset),
                       flags = load16(p + offset + 2);
        const uint32_t name_size = load32(p + offset + 4),
                       text_size = load32(p + offset + 8);
        const uint64_t bits = load64(p + offset + 12);
        const uint32_t reserved = load32(p + offset + 20);
        offset += WMFS_LOG_FIELD_SIZE;
        if (kind < 1 || kind > 5 || flags || reserved || !name_size ||
            name_size > WMFS_LOG_MAX_NAME_BYTES ||
            name_size + text_size > packet.size - offset ||
            (kind != 5 && text_size) || (kind == 5 && bits) ||
            !utf8(p + offset, name_size + text_size))
            return 1;
        if (kind == WMFS_LOG_WIRE_BOOLEAN && bits > 1)
            return 1;
        fields[i] = {kind,
                     bits,
                     {p + offset, name_size},
                     {p + offset + name_size, text_size}};
        offset += name_size + text_size;
    }
    return offset == packet.size ? 0 : 1;
}
