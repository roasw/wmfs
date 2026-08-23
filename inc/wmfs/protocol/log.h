#ifndef WMFS_PROTOCOL_LOG_H
#define WMFS_PROTOCOL_LOG_H

#include <stddef.h>
#include <stdint.h>

#define WMFS_LOG_MAGIC UINT64_C(0x31474f4c53464d57)
#define WMFS_LOG_ABI_MAJOR UINT16_C(1)
#define WMFS_LOG_ABI_MINOR UINT16_C(0)
#define WMFS_LOG_HEADER_SIZE UINT32_C(100)
#define WMFS_LOG_FIELD_SIZE UINT32_C(24)
#define WMFS_LOG_MAX_RECORD_BYTES UINT32_C(65536)
#define WMFS_LOG_MAX_FIELDS UINT32_C(32)
#define WMFS_LOG_MAX_MESSAGE_BYTES UINT32_C(32768)
#define WMFS_LOG_MAX_CATEGORY_BYTES UINT32_C(1024)
#define WMFS_LOG_MAX_NAME_BYTES UINT32_C(255)

typedef enum wmfs_log_record_flag_v1 {
    WMFS_LOG_RECORD_TRUNCATED = 1U << 0,
    WMFS_LOG_RECORD_SYNTHETIC = 1U << 1
} wmfs_log_record_flag_v1;

typedef enum wmfs_log_wire_field_kind_v1 {
    WMFS_LOG_WIRE_BOOLEAN = 1,
    WMFS_LOG_WIRE_INT64 = 2,
    WMFS_LOG_WIRE_UINT64 = 3,
    WMFS_LOG_WIRE_FLOAT64 = 4,
    WMFS_LOG_WIRE_TEXT = 5
} wmfs_log_wire_field_kind_v1;

/* Decoded views only. Records are encoded field-by-field little-endian. */
typedef struct wmfs_log_bytes_v1 {
    const uint8_t *data;
    size_t size;
} wmfs_log_bytes_v1;

typedef struct wmfs_log_record_v1 {
    uint32_t flags;
    uint32_t level;
    uint32_t field_count;
    uint64_t sequence;
    uint64_t time_ns;
    uint64_t session_id;
    uint64_t submission_id;
    uint64_t invocation_id;
    uint64_t operation_id;
    uint64_t dropped_before;
    wmfs_log_bytes_v1 category;
    wmfs_log_bytes_v1 message;
} wmfs_log_record_v1;

typedef struct wmfs_log_field_view_v1 {
    uint16_t kind;
    uint64_t bits;
    wmfs_log_bytes_v1 name;
    wmfs_log_bytes_v1 text;
} wmfs_log_field_view_v1;

/* Returns 0 on success and 1 for every malformed record. */
int wmfs_log_decode_v1(wmfs_log_bytes_v1 packet, wmfs_log_record_v1 *record,
                       wmfs_log_field_view_v1 *fields, size_t capacity);

#endif
