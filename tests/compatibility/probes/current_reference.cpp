#include <wmfs/reference_plugin.hpp>

static_assert(WMFS_REFERENCE_ABI_VERSION == 1, "plugin ABI version");
static_assert(WMFS_REFERENCE_CONFIGURATION_SCHEMA_VERSION == 1,
              "configuration schema version");
static_assert(WMFS_REFERENCE_HAS_INITIALIZE == 1, "initialize declaration");
static_assert(WMFS_REFERENCE_HAS_SHUTDOWN == 1, "shutdown declaration");
static_assert(wmfs::reference::configuration::Precision::fast !=
                  wmfs::reference::configuration::Precision::accurate,
              "configuration enum declaration");

typedef std::int32_t (*float_operation)(wmfs::reference::dtype_tag<float>,
                                        const wmfs_invocation_v1 *);

int wmfs_current_cpp_probe(const wmfs_plugin_api_v1 *api,
                           const wmfs_logger_v1 *native_logger) {
    wmfs::reference::logger log(native_logger);
    float_operation operation = &wmfs::reference::matmul_typed<float>;
    wmfs_initialize_v1 initialize = api->initialize;
    wmfs_shutdown_v1 shutdown = api->shutdown;
    wmfs::reference::dtype_tag<double> float64;
    wmfs::reference::dtype_tag<std::int64_t> int64;
    wmfs::reference::dtype_tag<std::uint8_t> uint8;

    if (log.enabled(WMFS_LOG_DEBUG))
        log.debug("gcc48", UINT64_C(5));
    (void)operation;
    (void)initialize;
    (void)shutdown;
    (void)float64;
    (void)int64;
    (void)uint8;
    return wmfs::reference::interface_fingerprint_sha256[0] == 0 ||
                   wmfs::reference::configuration_fingerprint_sha256[0] == 0 ||
                   wmfs::reference::configuration::threads_key[0] == '\0' ||
                   wmfs::reference::configuration::solver_algorithm_key[0] ==
                       '\0'
               ? 1
               : 0;
}
