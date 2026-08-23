#include <wmfs/reference_plugin.hpp>

static_assert(WMFS_REFERENCE_ABI_VERSION == 1, "plugin ABI version");
static_assert(wmfs::reference::matmul == 1, "operation declaration");
static_assert(wmfs::reference::nonzero == 6, "operation declaration");

const wmfs_plugin_api_v1 *wmfs_frozen_v1_cpp_probe(void) {
    return wmfs_plugin_get_api(WMFS_REFERENCE_ABI_VERSION);
}
