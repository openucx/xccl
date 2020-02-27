/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "tccl_team_lib.h"

static tccl_status_t tccl_create_team_context(tccl_team_lib_h lib,
                                              tccl_context_config_h config,
                                              tccl_context_h *team_ctx)
{
    return lib->create_team_context(lib, config, team_ctx);
}

static tccl_status_t find_tlib_by_name(tccl_lib_t *lib, const char *tlib_name,
                                       tccl_team_lib_t **tlib) {
    int i;
    for (i=0; i<lib->n_libs_opened; i++) {
        if (0 == strcmp(tlib_name, lib->libs[i]->name)) {
            *tlib = lib->libs[i];
            return TCCL_OK;
        }
    }
    *tlib = NULL;
    return TCCL_ERR_NO_ELEM;
}

tccl_status_t tccl_create_context(tccl_lib_h lib, tccl_context_config_t config,
                                  tccl_context_t **context)
{
    tccl_status_t status;
    if (config.field_mask & TCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME) {
        tccl_team_lib_t *tlib;
        if (TCCL_OK != (status = find_tlib_by_name(lib, config.team_lib_name, &tlib))) {
            return status;
        }
        return tccl_create_team_context(tlib, &config, context);
    } else {
        //TODO automatic team lib selection from the list of opened components with
        // prioritization
        return TCCL_ERR_NOT_IMPLEMENTED;
    }
    return TCCL_OK;
}

tccl_status_t tccl_destroy_context(tccl_context_h team_ctx)
{
    return team_ctx->lib->destroy_team_context(team_ctx);
}
