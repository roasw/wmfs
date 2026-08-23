#include <wmfs/plugin_abi.h>

typedef char abi_version_must_be_one[WMFS_PLUGIN_ABI_VERSION == 1 ? 1 : -1];
typedef char logger_must_be_declared[sizeof(wmfs_logger_v1) > 0 ? 1 : -1];

int wmfs_mode_neutral_c_abi_probe(void) {
    wmfs_tensor_v1 tensor = {0};
    tensor.dtype = WMFS_DTYPE_FLOAT64;
    return tensor.dtype == WMFS_DTYPE_FLOAT64 ? 0 : 1;
}
