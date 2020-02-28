/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <api/xccl.h>
#include <stdio.h>

int main(int argc, char **argv) {

    xccl_lib_config_t lib_config = {
        .field_mask = XCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = XCCL_USAGE_SW_COLLECTIVES |
                      XCCL_USAGE_HW_COLLECTIVES,
    };
    xccl_lib_h lib;
    xccl_lib_init(lib_config, &lib);

    xccl_context_config_t team_ctx_config = {
        .field_mask = XCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME |
                      XCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE |
                      XCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
        .team_lib_name   = "ucx",
        .thread_mode     = XCCL_LIB_THREAD_SINGLE,
        .completion_type = XCCL_TEAM_COMPLETION_BLOCKING,
    };
    xccl_context_h team_ctx;
    xccl_create_context(lib, team_ctx_config, &team_ctx);
    return 0;
}
