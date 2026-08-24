#ifndef WMFS_PROTOCOL_CONTROL_H
#define WMFS_PROTOCOL_CONTROL_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define WMFS_CONTROL_STATIC_ASSERT(condition, message)                         \
    static_assert(condition, message)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define WMFS_CONTROL_STATIC_ASSERT(condition, message)                         \
    _Static_assert(condition, message)
#else
#error "wmfs/protocol/control.h requires C11 or C++11"
#endif

#define WMFS_CONTROL_MAGIC UINT64_C(0x314c544353464d57)
#define WMFS_CONTROL_ABI_MAJOR UINT16_C(1)
#define WMFS_CONTROL_ABI_MINOR UINT16_C(0)
#define WMFS_CONTROL_FRAME_HEADER_SIZE UINT32_C(32)
#define WMFS_CONTROL_STARTUP_FIXED_SIZE UINT32_C(112)
#define WMFS_CONTROL_DESCRIPTOR_ROLE_SIZE UINT32_C(8)
#define WMFS_CONTROL_FD_BATCH_HEADER_SIZE UINT32_C(32)
#define WMFS_CONTROL_FD_ENTRY_SIZE UINT32_C(48)
#define WMFS_CONTROL_FD_ACK_FIXED_SIZE UINT32_C(32)
#define WMFS_CONTROL_MAX_CONFIG_BYTES UINT32_C(65536)
#define WMFS_CONTROL_MAX_PACKET_BYTES UINT32_C(69632)
#define WMFS_CONTROL_MAX_ERROR_BYTES UINT32_C(1024)
#define WMFS_CONTROL_MAX_DESCRIPTOR_ROLES UINT16_C(16)
#define WMFS_CONTROL_MAX_FD_ENTRIES UINT16_C(240)
#define WMFS_CONTROL_SHA256_SIZE UINT32_C(32)

typedef enum wmfs_control_frame_kind_v1 {
    WMFS_CONTROL_KIND_INVALID = 0,
    WMFS_CONTROL_STARTUP_REQUEST = 1,
    WMFS_CONTROL_STARTUP_RESPONSE = 2,
    WMFS_CONTROL_PING = 3,
    WMFS_CONTROL_PONG = 4,
    WMFS_CONTROL_SHUTDOWN = 5,
    WMFS_CONTROL_SHUTDOWN_ACK = 6,
    WMFS_CONTROL_ERROR_RESPONSE = 7,
    WMFS_CONTROL_FD_TRANSFER = 16,
    WMFS_CONTROL_FD_TRANSFER_ACK = 17
} wmfs_control_frame_kind_v1;

typedef enum wmfs_control_status_v1 {
    WMFS_CONTROL_STATUS_OK = 0,
    WMFS_CONTROL_STATUS_INVALID_ARGUMENT = 1,
    WMFS_CONTROL_STATUS_UNSUPPORTED = 2,
    WMFS_CONTROL_STATUS_IDENTITY_MISMATCH = 3,
    WMFS_CONTROL_STATUS_CONFIGURATION_REJECTED = 4,
    WMFS_CONTROL_STATUS_INTERNAL_ERROR = 5
} wmfs_control_status_v1;

typedef enum wmfs_control_capability_v1 {
    WMFS_CONTROL_CAPABILITY_NONE = 0,
    WMFS_CONTROL_CAPABILITY_COMMAND_RING = UINT64_C(1) << 0,
    WMFS_CONTROL_CAPABILITY_COMPLETION_RING = UINT64_C(1) << 1,
    WMFS_CONTROL_CAPABILITY_FD_CONTROL = UINT64_C(1) << 2,
    WMFS_CONTROL_CAPABILITY_CONFIGURATION = UINT64_C(1) << 3,
    WMFS_CONTROL_CAPABILITY_CENTRALIZED_LOGGING = UINT64_C(1) << 4,
    WMFS_CONTROL_CAPABILITY_WORKER_FILE_LOGGING = UINT64_C(1) << 5,
    WMFS_CONTROL_CAPABILITY_INITIALIZE = UINT64_C(1) << 6,
    WMFS_CONTROL_CAPABILITY_SHUTDOWN = UINT64_C(1) << 7
} wmfs_control_capability_v1;

typedef enum wmfs_control_log_mode_v1 {
    WMFS_CONTROL_LOG_DISABLED = 0,
    WMFS_CONTROL_LOG_CENTRALIZED = 1,
    WMFS_CONTROL_LOG_WORKER_FILE = 2
} wmfs_control_log_mode_v1;

typedef enum wmfs_control_descriptor_role_v1 {
    WMFS_CONTROL_DESCRIPTOR_INVALID = 0,
    WMFS_CONTROL_DESCRIPTOR_COMMAND_RING = 1,
    WMFS_CONTROL_DESCRIPTOR_COMMAND_DATA_EVENT = 2,
    WMFS_CONTROL_DESCRIPTOR_COMMAND_SPACE_EVENT = 3,
    WMFS_CONTROL_DESCRIPTOR_COMPLETION_RING = 4,
    WMFS_CONTROL_DESCRIPTOR_COMPLETION_DATA_EVENT = 5,
    WMFS_CONTROL_DESCRIPTOR_COMPLETION_SPACE_EVENT = 6,
    WMFS_CONTROL_DESCRIPTOR_FD_CONTROL = 7,
    WMFS_CONTROL_DESCRIPTOR_LOG = 8
} wmfs_control_descriptor_role_v1;

typedef enum wmfs_control_fd_entry_kind_v1 {
    WMFS_CONTROL_FD_ENTRY_INVALID = 0,
    WMFS_CONTROL_FD_ENTRY_MAP = 1,
    WMFS_CONTROL_FD_ENTRY_RETIRE = 2
} wmfs_control_fd_entry_kind_v1;

typedef enum wmfs_control_fd_flag_v1 {
    WMFS_CONTROL_FD_FLAG_NONE = 0,
    WMFS_CONTROL_FD_FLAG_WRITABLE = 1U << 0,
    WMFS_CONTROL_FD_FLAG_ARENA = 1U << 1
} wmfs_control_fd_flag_v1;

typedef enum wmfs_control_fd_batch_flag_v1 {
    WMFS_CONTROL_FD_BATCH_FLAG_NONE = 0,
    /* The receiver applies every ordered entry or invalidates the session. */
    WMFS_CONTROL_FD_BATCH_FLAG_TRANSACTIONAL = 1U << 0
} wmfs_control_fd_batch_flag_v1;

/* These records document decoded fields and wire offsets. Codecs must use
 * explicit little-endian loads/stores; copying these structs is not an ABI. */
typedef struct wmfs_control_frame_header_v1 {
    uint64_t magic;
    uint16_t abi_major;
    uint16_t abi_minor;
    uint16_t kind;
    uint16_t flags;
    uint32_t header_size;
    uint32_t payload_size;
    uint64_t request_id;
} wmfs_control_frame_header_v1;

typedef struct wmfs_control_descriptor_role_record_v1 {
    uint16_t role;
    uint16_t flags;
    uint32_t reserved;
} wmfs_control_descriptor_role_record_v1;

typedef struct wmfs_control_startup_v1 {
    uint64_t session_generation;
    uint8_t interface_fingerprint[32];
    uint8_t configuration_fingerprint[32];
    uint64_t metadata_fingerprint;
    uint64_t capabilities;
    uint32_t operation_count;
    uint32_t protocol_version;
    uint32_t configuration_schema_version;
    uint32_t config_length;
    uint16_t descriptor_count;
    uint16_t log_mode;
    uint32_t status;
} wmfs_control_startup_v1;

typedef struct wmfs_control_fd_batch_v1 {
    uint64_t transfer_id;
    uint64_t session_generation;
    uint16_t entry_count;
    uint16_t fd_count;
    uint32_t flags;
    uint64_t reserved;
} wmfs_control_fd_batch_v1;

typedef struct wmfs_control_fd_entry_v1 {
    uint32_t kind;
    uint32_t flags;
    uint64_t buffer_id;
    uint64_t generation;
    uint64_t allocation_id;
    uint64_t invocation_id;
    uint64_t byte_length;
} wmfs_control_fd_entry_v1;

typedef struct wmfs_control_fd_ack_v1 {
    uint64_t transfer_id;
    uint64_t session_generation;
    uint32_t status;
    uint32_t flags;
    uint32_t error_length;
    uint32_t reserved;
} wmfs_control_fd_ack_v1;

typedef struct wmfs_control_error_v1 {
    uint32_t status;
    uint32_t error_length;
} wmfs_control_error_v1;

typedef struct wmfs_control_bytes_v1 {
    const uint8_t *data;
    size_t size;
} wmfs_control_bytes_v1;

typedef struct wmfs_control_mutable_bytes_v1 {
    uint8_t *data;
    size_t capacity;
    size_t size;
} wmfs_control_mutable_bytes_v1;

typedef struct wmfs_control_startup_view_v1 {
    uint64_t request_id;
    wmfs_control_startup_v1 startup;
    const wmfs_control_descriptor_role_record_v1 *roles;
    wmfs_control_bytes_v1 config;
} wmfs_control_startup_view_v1;

typedef struct wmfs_control_fd_batch_view_v1 {
    uint64_t request_id;
    wmfs_control_fd_batch_v1 batch;
    const wmfs_control_fd_entry_v1 *entries;
} wmfs_control_fd_batch_view_v1;

#ifdef __cplusplus
extern "C" {
#endif

int32_t wmfs_control_encode_startup_v1(
    uint16_t kind, uint64_t request_id, const wmfs_control_startup_v1 *startup,
    const wmfs_control_descriptor_role_record_v1 *roles,
    wmfs_control_bytes_v1 config, wmfs_control_mutable_bytes_v1 *output);
int32_t wmfs_control_decode_startup_v1(
    wmfs_control_bytes_v1 packet, uint16_t expected_kind,
    wmfs_control_startup_view_v1 *output,
    wmfs_control_descriptor_role_record_v1 *role_storage, size_t role_capacity);
int32_t wmfs_control_encode_fd_batch_v1(uint64_t request_id,
                                        const wmfs_control_fd_batch_v1 *batch,
                                        const wmfs_control_fd_entry_v1 *entries,
                                        wmfs_control_mutable_bytes_v1 *output);
int32_t wmfs_control_decode_fd_batch_v1(wmfs_control_bytes_v1 packet,
                                        wmfs_control_fd_batch_view_v1 *output,
                                        wmfs_control_fd_entry_v1 *entry_storage,
                                        size_t entry_capacity);
int32_t wmfs_control_encode_fd_ack_v1(uint64_t request_id,
                                      const wmfs_control_fd_ack_v1 *ack,
                                      wmfs_control_bytes_v1 error,
                                      wmfs_control_mutable_bytes_v1 *output);
int32_t wmfs_control_decode_fd_ack_v1(wmfs_control_bytes_v1 packet,
                                      uint64_t *request_id,
                                      wmfs_control_fd_ack_v1 *ack,
                                      wmfs_control_bytes_v1 *error);
int32_t wmfs_control_encode_error_v1(uint64_t request_id,
                                     const wmfs_control_error_v1 *error,
                                     wmfs_control_bytes_v1 message,
                                     wmfs_control_mutable_bytes_v1 *output);
int32_t wmfs_control_decode_error_v1(wmfs_control_bytes_v1 packet,
                                     uint64_t *request_id,
                                     wmfs_control_error_v1 *error,
                                     wmfs_control_bytes_v1 *message);
int32_t wmfs_control_encode_empty_v1(uint16_t kind, uint64_t request_id,
                                     wmfs_control_mutable_bytes_v1 *output);
int32_t wmfs_control_decode_empty_v1(wmfs_control_bytes_v1 packet,
                                     uint16_t expected_kind,
                                     uint64_t *request_id);

#ifdef __cplusplus
}
#endif

WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_frame_header_v1) == 32,
                           "control frame header layout");
WMFS_CONTROL_STATIC_ASSERT(offsetof(wmfs_control_frame_header_v1,
                                    payload_size) == 20,
                           "control payload size offset");
WMFS_CONTROL_STATIC_ASSERT(offsetof(wmfs_control_frame_header_v1, request_id) ==
                               24,
                           "control request ID offset");
WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_descriptor_role_record_v1) == 8,
                           "descriptor role layout");
WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_startup_v1) == 112,
                           "startup layout");
WMFS_CONTROL_STATIC_ASSERT(offsetof(wmfs_control_startup_v1, config_length) ==
                               100,
                           "startup config length offset");
WMFS_CONTROL_STATIC_ASSERT(offsetof(wmfs_control_startup_v1,
                                    descriptor_count) == 104,
                           "startup descriptor count offset");
WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_fd_batch_v1) == 32,
                           "FD batch layout");
WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_fd_entry_v1) == 48,
                           "FD entry layout");
WMFS_CONTROL_STATIC_ASSERT(offsetof(wmfs_control_fd_entry_v1, generation) == 16,
                           "FD generation offset");
WMFS_CONTROL_STATIC_ASSERT(offsetof(wmfs_control_fd_entry_v1, invocation_id) ==
                               32,
                           "FD invocation offset");
WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_fd_ack_v1) == 32,
                           "FD acknowledgement layout");
WMFS_CONTROL_STATIC_ASSERT(sizeof(wmfs_control_error_v1) == 8,
                           "error response layout");

#undef WMFS_CONTROL_STATIC_ASSERT
#endif
