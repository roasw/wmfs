#include <wmfs/mode_neutral_plugin.hpp>

static_assert(WMFS_MODE_NEUTRAL_ABI_VERSION == 1, "plugin ABI version");
static_assert(WMFS_MODE_NEUTRAL_HAS_INITIALIZE == 0, "no initialize hook");
static_assert(WMFS_MODE_NEUTRAL_HAS_SHUTDOWN == 0, "no shutdown hook");

typedef std::int32_t (*scale_operation)(wmfs::mode_neutral::dtype_tag<double>,
                                        const wmfs_invocation_v1 *);

int wmfs_mode_neutral_cpp_probe(void) {
    scale_operation operation = &wmfs::mode_neutral::scale_typed<double>;
    wmfs::mode_neutral::logger logger;
    (void)operation;
    return logger.native_handle() == 0 &&
                   wmfs::mode_neutral::interface_fingerprint_sha256[0] != 0
               ? 0
               : 1;
}
