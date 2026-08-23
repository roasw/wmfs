#include <wmfs/plugin_abi.h>

_Static_assert(WMFS_PLUGIN_ABI_VERSION == 1, "plugin ABI version");
_Static_assert(WMFS_PLUGIN_MAX_LOG_FIELDS == 32, "logger field bound");
_Static_assert(sizeof(wmfs_json_view_v1) == sizeof(wmfs_text_view_v1),
               "JSON view ABI");

static uint8_t enabled(void *context, uint32_t level) {
    (void)context;
    return level >= WMFS_LOG_INFO;
}

static void write_log(void *context, uint32_t level, wmfs_text_view_v1 message,
                      wmfs_text_view_v1 category,
                      const wmfs_log_field_v1 *fields, uint32_t field_count) {
    (void)context;
    (void)level;
    (void)message;
    (void)category;
    (void)fields;
    (void)field_count;
}

int wmfs_current_c_abi_probe(void) {
    wmfs_logger_v1 logger = {sizeof(wmfs_logger_v1), 0, 0, enabled, write_log};
    wmfs_initialize_args_v1 initialize = {
        sizeof(wmfs_initialize_args_v1), 0, 0, {0, 0}, logger, 0};
    wmfs_log_field_v1 field = {
        sizeof(wmfs_log_field_v1), WMFS_LOG_FIELD_UINT64, {0, 0}, 0, {0, 0}};

    return initialize.logger.enabled(0, WMFS_LOG_INFO) &&
                   field.kind == WMFS_LOG_FIELD_UINT64
               ? 0
               : 1;
}
