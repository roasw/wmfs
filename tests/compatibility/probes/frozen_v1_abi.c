#include <wmfs/plugin_abi.h>

typedef char abi_version_must_be_one[WMFS_PLUGIN_ABI_VERSION == 1 ? 1 : -1];
typedef char
    invocation_must_be_declared[sizeof(wmfs_invocation_v1) > 0 ? 1 : -1];

int wmfs_frozen_v1_c_abi_probe(void) {
    wmfs_plugin_api_v1 api = {0};
    api.abi_version = WMFS_PLUGIN_ABI_VERSION;
    return api.abi_version == 1 ? 0 : 1;
}
