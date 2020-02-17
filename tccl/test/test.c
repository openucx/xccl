/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "src/api/tccl.h"
#include <stdio.h>

int query_lib_params(tccl_team_lib_h team_lib) {
    tccl_team_lib_attr_t team_lib_attr;
    team_lib_attr.field_mask = TCCL_ATTR_FIELD_CONTEXT_CREATE_MODE;
    tccl_team_lib_query(team_lib, &team_lib_attr);
    printf("Context create mode: %d(%s)\n", team_lib_attr.context_create_mode,
        team_lib_attr.context_create_mode ? "global": "local");
}

int main(int argc, char **argv) {

    tccl_team_lib_params_t lib_params = {
        .tccl_model_type = TCCL_MODEL_TYPE_MPI,
        .team_world_size = 8,
        .team_lib_name = "ucx",
        .ucp_context = NULL
    };
    tccl_team_lib_h lib;
    tccl_team_lib_init(&lib_params, &lib);
    query_lib_params(lib);

    tccl_team_context_config_t team_ctx_config = {
        .thread_support = TCCL_THREAD_MODE_PRIVATE,
        .completion_type = TCCL_TEAM_COMPLETION_BLOCKING
    };
    tccl_team_context_h team_ctx;
    tccl_create_team_context(lib, &team_ctx_config, &team_ctx);

    return 0;
}
