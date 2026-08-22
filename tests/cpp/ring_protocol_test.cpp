#include "wmfs/protocol/ring.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_constants_and_layout() {
    require(WMFS_RING_HEADER_SIZE == sizeof(wmfs_ring_header_v1),
            "header size constant does not match the type");
    require(WMFS_RING_RECORD_SIZE == sizeof(wmfs_ring_record_v1),
            "record size constant does not match the type");
    require(WMFS_RING_MAX_TENSORS == 16 && WMFS_RING_MAX_RANK == 16,
            "tensor bounds changed");
    require(WMFS_RING_MAX_SCALARS == 16 && WMFS_RING_MAX_SCALAR_TEXT == 256,
            "scalar bounds changed");
    require(WMFS_RING_MAX_PLANNED_OUTPUTS == 16,
            "planned output bound changed");
    require(WMFS_RING_MAX_ERROR_TYPE == 64 &&
                WMFS_RING_MAX_ERROR_MESSAGE == 1024,
            "error bounds changed");
    require(offsetof(wmfs_ring_header_v1, producer) == 64,
            "producer is not on its assigned cache line");
    require(offsetof(wmfs_ring_header_v1, consumer) == 128,
            "consumer is not on its assigned cache line");
    require(offsetof(wmfs_ring_record_v1, tensors) == 128, "tensor area moved");
    require(offsetof(wmfs_ring_record_v1, error) == 11840, "error area moved");
}

void test_basic_record_roundtrip() {
    wmfs_ring_command_v1 command;
    std::memset(&command, 0, sizeof(command));
    command.kind = WMFS_RING_COMMAND_INVOKE;
    command.flags = WMFS_RING_RECORD_FLAG_PROFILE;
    command.session_generation = UINT64_C(7);
    command.submission_id = UINT64_C(41);
    command.invocation_id = UINT64_C(43);
    command.operation_id = UINT32_C(47);
    command.tensor_count = 1;
    command.scalar_count = 1;
    command.tensors[0].buffer_id = UINT64_C(53);
    command.tensors[0].buffer_generation = UINT64_C(2);
    command.tensors[0].byte_length = UINT64_C(48);
    command.tensors[0].dtype = WMFS_RING_DTYPE_FLOAT64;
    command.tensors[0].rank = 2;
    command.tensors[0].kind = WMFS_RING_TENSOR_INPUT;
    command.tensors[0].shape[0] = 2;
    command.tensors[0].shape[1] = 3;
    command.tensors[0].strides[0] = 3;
    command.tensors[0].strides[1] = 1;
    command.scalars[0].parameter_index = 1;
    command.scalars[0].kind = WMFS_RING_SCALAR_TEXT;
    command.scalars[0].text_length = 3;
    std::memcpy(command.scalars[0].text, "svd", 3);

    uint8_t bytes[WMFS_RING_RECORD_SIZE];
    std::memcpy(bytes, &command, sizeof(command));
    wmfs_ring_command_v1 decoded;
    std::memcpy(&decoded, bytes, sizeof(decoded));

    require(decoded.kind == WMFS_RING_COMMAND_INVOKE,
            "command kind did not round trip");
    require(decoded.session_generation == 7 && decoded.submission_id == 41 &&
                decoded.invocation_id == 43 && decoded.operation_id == 47,
            "record identifiers did not round trip");
    require(decoded.tensors[0].shape[1] == 3 &&
                decoded.tensors[0].strides[0] == 3,
            "tensor descriptor did not round trip");
    require(decoded.scalars[0].text_length == 3 &&
                std::memcmp(decoded.scalars[0].text, "svd", 3) == 0,
            "scalar did not round trip");

    wmfs_ring_completion_v1 completion;
    std::memset(&completion, 0, sizeof(completion));
    completion.kind = WMFS_RING_COMPLETION_INVOKE;
    completion.status = WMFS_RING_STATUS_OPERATION_ERROR;
    completion.session_generation = decoded.session_generation;
    completion.submission_id = decoded.submission_id;
    completion.invocation_id = decoded.invocation_id;
    completion.operation_id = decoded.operation_id;
    completion.error.type_length = 10;
    completion.error.message_length = 12;
    std::memcpy(completion.error.type, "ValueError", 10);
    std::memcpy(completion.error.message, "bad argument", 12);
    std::memcpy(bytes, &completion, sizeof(completion));
    wmfs_ring_completion_v1 decoded_completion;
    std::memcpy(&decoded_completion, bytes, sizeof(decoded_completion));
    require(decoded_completion.kind == WMFS_RING_COMPLETION_INVOKE &&
                decoded_completion.status == WMFS_RING_STATUS_OPERATION_ERROR,
            "completion status did not round trip");
    require(decoded_completion.error.type[0] == 'V' &&
                decoded_completion.error.message[11] == 't',
            "completion error data did not round trip");
}

} // namespace

int main() {
    test_constants_and_layout();
    test_basic_record_roundtrip();
}
