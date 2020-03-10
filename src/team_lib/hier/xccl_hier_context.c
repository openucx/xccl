/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_hier_context.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

static xccl_status_t init_env_params(xccl_hier_context_t *ctx)
{
    char *var;
    /* Just quick simple getenv to have smth working.
       TODO We need a parameter registration framework. */

    var = getenv("XCCL_HIER_ENABLE_SHARP");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->tls[XCCL_TL_SHARP].enabled = 1;
    } else {
        ctx->tls[XCCL_TL_SHARP].enabled = 0;
    }

    var = getenv("XCCL_HIER_ENABLE_SHMSEG");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->tls[XCCL_TL_SHMSEG].enabled = 1;
    } else {
        ctx->tls[XCCL_TL_SHMSEG].enabled = 0;
    }

    var = getenv("XCCL_HIER_ENABLE_VMC");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->tls[XCCL_TL_VMC].enabled = 1;
    } else {
        ctx->tls[XCCL_TL_VMC].enabled = 0;
    }

    var = getenv("XCCL_BCAST_PIPELINE_THRESH");
    if (var) {
        if (0 == strcmp("inf", var) || 0 == strcmp("INF", var)) {
            ctx->bcast_pipeline_thresh = SIZE_MAX;
        } else {
            ctx->bcast_pipeline_thresh = (size_t)atoi(var);
        }
    } else {
        ctx->bcast_pipeline_thresh = SIZE_MAX;
    }

    var = getenv("XCCL_BCAST_PIPELINE_DEPTH");
    if (var) {
        ctx->bcast_pipeline_depth = atoi(var);
    } else {
        ctx->bcast_pipeline_depth = 1;
    }

    var = getenv("XCCL_BCAST_SM_GET");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->use_sm_get_bcast = 1;
    } else {
        ctx->use_sm_get_bcast = 0;
    }

    var = getenv("XCCL_BCAST_SM_GET_THRESH");
    if (var) {
        if (0 == strcmp("inf", var) || 0 == strcmp("INF", var)) {
            ctx->bcast_sm_get_thresh = SIZE_MAX;
        } else {
            ctx->bcast_sm_get_thresh = (size_t)atoi(var);
        }
    } else {
        ctx->bcast_sm_get_thresh = SIZE_MAX;
    }
    return XCCL_OK;
}

static xccl_status_t
xccl_hier_init_tl(xccl_hier_context_t *ctx, xccl_tl_id_t tl_id,
                  xccl_oob_collectives_t oob) {
    if (!ctx->tls[tl_id].enabled) {
        return XCCL_OK;
    }
    xccl_params_t params = {
        .field_mask = XCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = XCCL_USAGE_SW_COLLECTIVES |
                      XCCL_USAGE_HW_COLLECTIVES,
    };
    xccl_config_t config = {
        .ctx_config = {
            .field_mask = XCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE |
                          XCCL_CONTEXT_CONFIG_FIELD_OOB |
                          XCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
            .thread_mode     = XCCL_LIB_THREAD_SINGLE,
            .completion_type = XCCL_TEAM_COMPLETION_BLOCKING,
            .oob = oob,
        },
        .tls = xccl_tl_str(tl_id),
    };

    if (XCCL_OK != xccl_init(&params, &config,
                             &ctx->tls[tl_id].xccl_ctx)) {
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

static int compare_proc_data(const void* a, const void* b) {
    const xccl_hier_proc_data_t *d1 = (const xccl_hier_proc_data_t*)a;
    const xccl_hier_proc_data_t *d2 = (const xccl_hier_proc_data_t*)b;
    if (d1->node_hash != d2->node_hash) {
        return d1->node_hash > d2->node_hash ? 1 : -1;
    } else {
        return d1->socketid - d2->socketid;
    }
}

static void compute_layout(xccl_hier_context_t *ctx) {
    int ctx_size = ctx->super.cfg->oob.size;
    xccl_hier_proc_data_t *sorted = (xccl_hier_proc_data_t*)
        malloc(ctx_size*sizeof(xccl_hier_proc_data_t));
    memcpy(sorted, ctx->procs, ctx_size*sizeof(xccl_hier_proc_data_t));
    qsort(sorted, ctx_size, sizeof(xccl_hier_proc_data_t), compare_proc_data);
    unsigned long current_hash = sorted[0].node_hash;
    int current_ppn = 1;
    int min_ppn = INT_MAX;
    int max_ppn = 0;
    int nnodes = 1;
    int i, j;
    for (i=1; i<ctx_size; i++) {
        unsigned long hash = sorted[i].node_hash;
        if (hash != current_hash) {
            for (j=0; j<ctx_size; j++) {
                if (ctx->procs[j].node_hash == current_hash) {
                    ctx->procs[j].node_id = nnodes - 1;
                }
            }
            if (current_ppn > max_ppn) max_ppn = current_ppn;
            if (current_ppn < min_ppn) min_ppn = current_ppn;
            nnodes++;
            current_hash = hash;
            current_ppn = 1;
        } else {
            current_ppn++;
        }
    }
    for (j=0; j<ctx_size; j++) {
        if (ctx->procs[j].node_hash == current_hash) {
            ctx->procs[j].node_id = nnodes - 1;
        }
    }

    if (current_ppn > max_ppn) max_ppn = current_ppn;
    if (current_ppn < min_ppn) min_ppn = current_ppn;
    free(sorted);
    ctx->nnodes = nnodes;
    ctx->min_ppn = min_ppn;
    ctx->max_ppn = max_ppn;
}

xccl_status_t xccl_hier_create_context(xccl_team_lib_t *lib, xccl_context_config_t *config,
                                       xccl_tl_context_t **context)
{
    xccl_hier_context_t *ctx =
        (xccl_hier_context_t *)malloc(sizeof(*ctx));
    int i;
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);
    ctx->procs = (xccl_hier_proc_data_t*)malloc(
        config->oob.size*sizeof(xccl_hier_proc_data_t));
    ctx->local_proc.socketid = xccl_local_process_info()->socketid;
    ctx->local_proc.node_hash = xccl_local_process_info()->node_hash;
    memset(ctx->tls, 0, sizeof(ctx->tls));
    *context = NULL;

    for (i=0; i<XCCL_TL_LAST; i++) {
        ctx->tls[i].enabled = 1;
    }
    /* Disable recursion */
    ctx->tls[XCCL_TL_HIER].enabled = 0;
    init_env_params(ctx);

    for (i=0; i<XCCL_TL_LAST; i++) {
        if (XCCL_OK != xccl_hier_init_tl(ctx, i, config->oob)) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }

    xccl_oob_allgather(&ctx->local_proc, ctx->procs,
                       sizeof(xccl_hier_proc_data_t), &config->oob);
    compute_layout(ctx);
    *context = &ctx->super;
    return XCCL_OK;
}

xccl_status_t xccl_hier_destroy_context(xccl_tl_context_t *team_context)
{
    xccl_hier_context_t    *ctx = xccl_derived_of(team_context, xccl_hier_context_t);
    xccl_oob_collectives_t *oob = &team_context->cfg->oob;
    int i;
    for (i=0; i<XCCL_TL_LAST; i++) {
        if (ctx->tls[i].enabled) {
            assert(ctx->tls[i].xccl_ctx);
            xccl_cleanup(ctx->tls[i].xccl_ctx);
        }
    }
    free(ctx->procs);
    free(ctx);
    return XCCL_OK;
}

xccl_status_t xccl_hier_context_progress(xccl_tl_context_t *team_context)
{
    xccl_hier_context_t *ctx = xccl_derived_of(team_context, xccl_hier_context_t);
    int i;
    xccl_context_t *tl_ctx;
    for (i=0; i<XCCL_TL_LAST; i++) {
        if (ctx->tls[i].enabled) {
            tl_ctx = ctx->tls[i].xccl_ctx;
            assert(tl_ctx);
            xccl_context_progress(tl_ctx);
        }
    }
    return XCCL_OK;
}
