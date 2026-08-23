#include <stddef.h>
#include <wmfs/protocol/control.h>

int main(void) {
    return offsetof(wmfs_control_startup_v1, config_length) == 100 &&
                   offsetof(wmfs_control_fd_entry_v1, invocation_id) == 32
               ? 0
               : 1;
}
