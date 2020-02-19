/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <api/tccl.h>
#include <stdio.h>

int main(int argc, char **argv) {

    tccl_lib_config_t lib_config = {
        .field_mask = TCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = TCCL_USAGE_SW_COLLECTIVES |
                      TCCL_USAGE_HW_COLLECTIVES,
    };
    tccl_lib_h lib;
    tccl_lib_init(lib_config, &lib);

    tccl_context_config_t team_ctx_config = {
        .field_mask = TCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME |
                      TCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE |
                      TCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
        .team_lib_name   = "ucx",
        .thread_mode     = TCCL_LIB_THREAD_SINGLE,
        .completion_type = TCCL_TEAM_COMPLETION_BLOCKING,
    };
    tccl_context_h team_ctx;
    tccl_create_context(lib, team_ctx_config, &team_ctx);
    return 0;
}
