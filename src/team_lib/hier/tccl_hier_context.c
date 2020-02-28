/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "tccl_hier_context.h"
#include <stdlib.h>
#include <stdio.h>

static tccl_status_t init_env_params(tccl_hier_context_t *ctx)
{
    char *var;
    /* Just quick simple getenv to have smth working.
       TODO We need a parameter registration framework. */

    var = getenv("TCCL_HIER_ENABLE_SHARP");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->tls[TCCL_TL_SHARP].enabled = 1;
    } else {
        ctx->tls[TCCL_TL_SHARP].enabled = 0;
    }

    var = getenv("TCCL_HIER__ENABLE_SHMSEG");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->tls[TCCL_TL_SHMSEG].enabled = 1;
    } else {
        ctx->tls[TCCL_TL_SHMSEG].enabled = 0;
    }

    var = getenv("TCCL_HIER_ENABLE_VMC");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->tls[TCCL_TL_VMC].enabled = 1;
    } else {
        ctx->tls[TCCL_TL_VMC].enabled = 0;
    }

    return TCCL_OK;
}

static tccl_status_t
tccl_hier_init_tl(tccl_hier_context_t *ctx, tccl_tl_id_t tl_id,
                  tccl_oob_collectives_t oob) {
    if (!ctx->tls[tl_id].enabled) {
        return TCCL_OK;
    }
    tccl_params_t params = {
        .field_mask = TCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = TCCL_USAGE_SW_COLLECTIVES |
                      TCCL_USAGE_HW_COLLECTIVES,
    };
    tccl_config_t config = {
        .ctx_config = {
            .field_mask = TCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE |
                          TCCL_CONTEXT_CONFIG_FIELD_OOB |
                          TCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
            .thread_mode     = TCCL_LIB_THREAD_SINGLE,
            .completion_type = TCCL_TEAM_COMPLETION_BLOCKING,
            .oob = oob,
        },
        .tls = tccl_tl_str(tl_id),
    };

    if (TCCL_OK != tccl_init(&params, &config,
                             &ctx->tls[tl_id].tccl_ctx)) {
        return TCCL_ERR_NO_MESSAGE;
    }
    return TCCL_OK;
}

tccl_status_t tccl_hier_create_context(tccl_team_lib_t *lib, tccl_context_config_t *config,
                                       tccl_tl_context_t **context)
{
    tccl_hier_context_t *ctx =
        (tccl_hier_context_t *)malloc(sizeof(*ctx));
    int i;
    TCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);
    ctx->procs = (tccl_hier_proc_data_t*)malloc(
        config->oob.size*sizeof(tccl_hier_proc_data_t));
    ctx->local_proc.socketid = tccl_local_proc_info.socketid;
    memset(ctx->tls, 0, sizeof(ctx->tls));
    *context = NULL;

    for (i=0; i<TCCL_TL_LAST; i++) {
        ctx->tls[i].enabled = 1;
    }
    /* Disable recursion */
    ctx->tls[TCCL_TL_HIER].enabled = 0;
    init_env_params(ctx);

    for (i=0; i<TCCL_TL_LAST; i++) {
        if (TCCL_OK != tccl_hier_init_tl(ctx, i, config->oob)) {
            return TCCL_ERR_NO_MESSAGE;
        }
    }
    *context = &ctx->super;
    return TCCL_OK;
}

tccl_status_t tccl_hier_destroy_context(tccl_tl_context_t *team_context)
{
    tccl_hier_context_t    *ctx = tccl_derived_of(team_context, tccl_hier_context_t);
    tccl_oob_collectives_t *oob = &team_context->cfg->oob;
    int i;
    for (i=0; i<TCCL_TL_LAST; i++) {
        if (ctx->tls[i].enabled) {
            assert(ctx->tls[i].tccl_ctx);
            tccl_cleanup(ctx->tls[i].tccl_ctx);
        }
    }
    free(ctx->procs);
    free(ctx);
    return TCCL_OK;
}
