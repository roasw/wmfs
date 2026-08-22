#ifndef WMFS_PROTOCOL_RING_H
#define WMFS_PROTOCOL_RING_H

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
#define WMFS_RING_ALIGNAS(value) alignas(value)
#define WMFS_RING_STATIC_ASSERT(condition, message)                            \
    static_assert(condition, message)
#define WMFS_RING_ALIGNOF(type) alignof(type)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define WMFS_RING_ALIGNAS(value) _Alignas(value)
#define WMFS_RING_STATIC_ASSERT(condition, message)                            \
    _Static_assert(condition, message)
#define WMFS_RING_ALIGNOF(type) _Alignof(type)
#else
#error "wmfs/protocol/ring.h requires C11 or C++11"
#endif

#define WMFS_RING_MAGIC UINT64_C(0x31474e5253464d57)
#define WMFS_RING_ABI_MAJOR UINT32_C(1)
#define WMFS_RING_ABI_MINOR UINT32_C(1)
#define WMFS_RING_HEADER_SIZE UINT32_C(4096)
#define WMFS_RING_RECORD_SIZE UINT32_C(16384)
#define WMFS_RING_CACHE_LINE_SIZE UINT32_C(64)

#define WMFS_RING_MAX_TENSORS UINT32_C(16)
#define WMFS_RING_MAX_RANK UINT32_C(16)
#define WMFS_RING_MAX_SCALARS UINT32_C(16)
#define WMFS_RING_MAX_SCALAR_TEXT UINT32_C(256)
#define WMFS_RING_MAX_PLANNED_OUTPUTS UINT32_C(16)
#define WMFS_RING_MAX_ERROR_TYPE UINT32_C(64)
#define WMFS_RING_MAX_ERROR_MESSAGE UINT32_C(1024)

typedef enum wmfs_ring_header_flag_v1 {
    WMFS_RING_HEADER_FLAG_NONE = 0,
    WMFS_RING_HEADER_FLAG_INITIALIZED = 1U << 0,
    WMFS_RING_HEADER_FLAG_CLOSED = 1U << 1
} wmfs_ring_header_flag_v1;

typedef enum wmfs_ring_record_kind_v1 {
    WMFS_RING_RECORD_KIND_INVALID = 0,
    WMFS_RING_COMMAND_INVOKE = 1,
    WMFS_RING_COMMAND_PLAN_OUTPUTS = 2,
    WMFS_RING_COMMAND_PING = 3,
    WMFS_RING_COMMAND_SHUTDOWN = 4,
    WMFS_RING_COMPLETION_INVOKE = 257,
    WMFS_RING_COMPLETION_PLAN_OUTPUTS = 258,
    WMFS_RING_COMPLETION_PONG = 259,
    WMFS_RING_COMPLETION_SHUTDOWN = 260
} wmfs_ring_record_kind_v1;

typedef enum wmfs_ring_status_v1 {
    WMFS_RING_STATUS_UNSET = 0,
    WMFS_RING_STATUS_OK = 1,
    WMFS_RING_STATUS_INVALID_ARGUMENT = 2,
    WMFS_RING_STATUS_UNSUPPORTED = 3,
    WMFS_RING_STATUS_OPERATION_ERROR = 4,
    WMFS_RING_STATUS_INTERNAL_ERROR = 5
} wmfs_ring_status_v1;

typedef enum wmfs_ring_record_flag_v1 {
    WMFS_RING_RECORD_FLAG_NONE = 0,
    WMFS_RING_RECORD_FLAG_PROFILE = 1U << 0
} wmfs_ring_record_flag_v1;

typedef enum wmfs_ring_tensor_kind_v1 {
    WMFS_RING_TENSOR_KIND_INVALID = 0,
    WMFS_RING_TENSOR_INPUT = 1,
    WMFS_RING_TENSOR_OUTPUT = 2
} wmfs_ring_tensor_kind_v1;

typedef enum wmfs_ring_tensor_flag_v1 {
    WMFS_RING_TENSOR_FLAG_NONE = 0,
    WMFS_RING_TENSOR_FLAG_WRITABLE = 1U << 0
} wmfs_ring_tensor_flag_v1;

typedef enum wmfs_ring_dtype_v1 {
    WMFS_RING_DTYPE_INVALID = 0,
    WMFS_RING_DTYPE_BOOL = 1,
    WMFS_RING_DTYPE_INT8 = 2,
    WMFS_RING_DTYPE_UINT8 = 3,
    WMFS_RING_DTYPE_INT16 = 4,
    WMFS_RING_DTYPE_INT32 = 5,
    WMFS_RING_DTYPE_INT64 = 6,
    WMFS_RING_DTYPE_FLOAT16 = 7,
    WMFS_RING_DTYPE_FLOAT32 = 8,
    WMFS_RING_DTYPE_FLOAT64 = 9,
    WMFS_RING_DTYPE_BFLOAT16 = 10
} wmfs_ring_dtype_v1;

typedef enum wmfs_ring_scalar_kind_v1 {
    WMFS_RING_SCALAR_KIND_INVALID = 0,
    WMFS_RING_SCALAR_BOOLEAN = 1,
    WMFS_RING_SCALAR_FLOAT64 = 2,
    WMFS_RING_SCALAR_INT64 = 3,
    WMFS_RING_SCALAR_TEXT = 4
} wmfs_ring_scalar_kind_v1;

typedef enum wmfs_ring_scalar_flag_v1 {
    WMFS_RING_SCALAR_FLAG_NONE = 0
} wmfs_ring_scalar_flag_v1;

typedef enum wmfs_ring_planned_output_flag_v1 {
    WMFS_RING_PLANNED_OUTPUT_FLAG_NONE = 0
} wmfs_ring_planned_output_flag_v1;

typedef enum wmfs_ring_error_flag_v1 {
    WMFS_RING_ERROR_FLAG_NONE = 0,
    WMFS_RING_ERROR_FLAG_TRUNCATED_TYPE = 1U << 0,
    WMFS_RING_ERROR_FLAG_TRUNCATED_MESSAGE = 1U << 1
} wmfs_ring_error_flag_v1;

typedef struct wmfs_ring_tensor_descriptor_v1 {
    uint64_t buffer_id;
    uint64_t buffer_generation;
    uint64_t allocation_id;
    uint64_t byte_offset;
    uint64_t byte_length;
    uint32_t dtype;
    uint16_t rank;
    uint16_t kind;
    uint32_t flags;
    uint16_t parameter_index;
    uint16_t reserved0;
    int64_t shape[16];
    int64_t strides[16];
} wmfs_ring_tensor_descriptor_v1;

typedef struct wmfs_ring_scalar_v1 {
    uint16_t parameter_index;
    uint16_t kind;
    uint32_t flags;
    uint64_t bits;
    uint32_t text_length;
    uint32_t reserved0;
    char text[256];
} wmfs_ring_scalar_v1;

typedef struct wmfs_ring_planned_output_v1 {
    uint32_t dtype;
    uint32_t flags;
    uint16_t rank;
    uint16_t output_index;
    uint32_t reserved0;
    int64_t shape[16];
} wmfs_ring_planned_output_v1;

typedef struct wmfs_ring_profile_v1 {
    uint64_t command_published_ns;
    uint64_t worker_dequeued_ns;
    uint64_t worker_started_ns;
    uint64_t worker_input_views_ns;
    uint64_t worker_output_views_ns;
    uint64_t worker_dispatch_ns;
    uint64_t worker_kernel_ns;
    uint64_t completion_published_ns;
} wmfs_ring_profile_v1;

typedef struct wmfs_ring_error_v1 {
    uint32_t type_length;
    uint32_t message_length;
    uint32_t flags;
    uint32_t reserved0;
    char type[64];
    char message[1024];
} wmfs_ring_error_v1;

typedef struct wmfs_ring_header_v1 {
    WMFS_RING_ALIGNAS(64) uint64_t magic;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t capacity;
    uint32_t flags;
    uint64_t session_generation;
    uint8_t reserved0[24];
    uint64_t producer;
    uint8_t reserved1[56];
    uint64_t consumer;
    uint8_t reserved2[3960];
} wmfs_ring_header_v1;

typedef struct wmfs_ring_record_v1 {
    WMFS_RING_ALIGNAS(64) uint32_t kind;
    uint32_t flags;
    uint64_t session_generation;
    uint64_t submission_id;
    uint64_t invocation_id;
    uint32_t operation_id;
    uint32_t status;
    uint16_t tensor_count;
    uint16_t scalar_count;
    uint16_t planned_output_count;
    uint16_t reserved0;
    uint32_t reserved1;
    uint8_t reserved_header[76];
    wmfs_ring_tensor_descriptor_v1 tensors[16];
    wmfs_ring_scalar_v1 scalars[16];
    wmfs_ring_planned_output_v1 planned_outputs[16];
    wmfs_ring_profile_v1 profile;
    wmfs_ring_error_v1 error;
    uint8_t reserved_tail[3312];
} wmfs_ring_record_v1;

typedef wmfs_ring_record_v1 wmfs_ring_command_v1;
typedef wmfs_ring_record_v1 wmfs_ring_completion_v1;

WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_tensor_descriptor_v1) == 312,
                        "tensor descriptor ABI size");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, buffer_id) ==
                            0,
                        "tensor buffer ID ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1,
                                 buffer_generation) == 8,
                        "tensor generation ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1,
                                 allocation_id) == 16,
                        "tensor allocation ID ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, byte_offset) ==
                            24,
                        "tensor byte offset ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, byte_length) ==
                            32,
                        "tensor byte length ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, dtype) == 40,
                        "tensor dtype ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, rank) == 44,
                        "tensor rank ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, kind) == 46,
                        "tensor kind ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, flags) == 48,
                        "tensor flags ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1,
                                 parameter_index) == 52,
                        "tensor parameter index ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, shape) == 56,
                        "tensor shape ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_tensor_descriptor_v1, strides) ==
                            184,
                        "tensor strides ABI offset");
WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_scalar_v1) == 280, "scalar ABI size");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_scalar_v1, bits) == 8,
                        "scalar bits ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_scalar_v1, text_length) == 16,
                        "scalar text length ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_scalar_v1, text) == 24,
                        "scalar text ABI offset");
WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_planned_output_v1) == 144,
                        "planned output ABI size");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_planned_output_v1, rank) == 8,
                        "planned output rank ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_planned_output_v1, shape) == 16,
                        "planned output shape ABI offset");
WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_profile_v1) == 64, "profile ABI size");
WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_error_v1) == 1104, "error ABI size");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_error_v1, type) == 16,
                        "error type ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_error_v1, message) == 80,
                        "error message ABI offset");

WMFS_RING_STATIC_ASSERT(WMFS_RING_ALIGNOF(wmfs_ring_header_v1) == 64,
                        "ring header ABI alignment");
WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_header_v1) == 4096,
                        "ring header ABI size");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_header_v1, magic) == 0,
                        "ring magic ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_header_v1, session_generation) == 32,
                        "ring generation ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_header_v1, producer) == 64,
                        "producer ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_header_v1, consumer) == 128,
                        "consumer ABI offset");

WMFS_RING_STATIC_ASSERT(WMFS_RING_ALIGNOF(wmfs_ring_record_v1) == 64,
                        "ring record ABI alignment");
WMFS_RING_STATIC_ASSERT(sizeof(wmfs_ring_record_v1) == 16384,
                        "ring record ABI size");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, kind) == 0,
                        "record kind ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, session_generation) == 8,
                        "record generation ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, submission_id) == 16,
                        "submission ID ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, invocation_id) == 24,
                        "invocation ID ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, operation_id) == 32,
                        "operation ID ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, status) == 36,
                        "record status ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, tensor_count) == 40,
                        "tensor count ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, scalar_count) == 42,
                        "scalar count ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, planned_output_count) ==
                            44,
                        "planned output count ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, tensors) == 128,
                        "tensor array ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, scalars) == 5120,
                        "scalar array ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, planned_outputs) == 9600,
                        "planned output array ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, profile) == 11904,
                        "profile ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, error) == 11968,
                        "error ABI offset");
WMFS_RING_STATIC_ASSERT(offsetof(wmfs_ring_record_v1, reserved_tail) == 13072,
                        "record tail ABI offset");

#undef WMFS_RING_ALIGNOF
#undef WMFS_RING_STATIC_ASSERT
#undef WMFS_RING_ALIGNAS

#endif
