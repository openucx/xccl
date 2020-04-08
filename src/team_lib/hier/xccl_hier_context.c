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
#include <ucs/sys/math.h>

static xccl_status_t
xccl_hier_init_tl(xccl_hier_context_t *ctx, xccl_tl_id_t tl_id,
                  xccl_oob_collectives_t oob, const char* prefix) {
    xccl_lib_h            lib;
    xccl_context_config_t *cfg;
    char env_prefix[128];

    if (!ctx->tls[tl_id].enabled) {
        return XCCL_OK;
    }
    xccl_lib_params_t lib_params = {
        .field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE,
        .team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                      XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    };

    xccl_context_params_t ctx_params = {
        .field_mask      = XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE |
                           XCCL_CONTEXT_PARAM_FIELD_TEAM_COMPLETION_TYPE |
                           XCCL_CONTEXT_PARAM_FIELD_OOB |
                           XCCL_CONTEXT_PARAM_FIELD_TLS,
        .thread_mode     = XCCL_THREAD_MODE_SINGLE,
        .completion_type = XCCL_TEAM_COMPLETION_TYPE_BLOCKING,
        .oob             = oob,
        .tls             = 1 << tl_id,
    };

    if (XCCL_OK != xccl_lib_init(&lib_params, NULL, &lib)) {
        return XCCL_ERR_NO_MESSAGE;
    }

    if (prefix != NULL) {
        snprintf(env_prefix, sizeof(env_prefix), "%s_HIER_%s",
                 prefix, xccl_tl_str(1 << tl_id));
    }
    else {
        snprintf(env_prefix, sizeof(env_prefix), "HIER_%s", xccl_tl_str(1 << tl_id));
    }

    xccl_context_config_read(lib, env_prefix, NULL, &cfg);
    if (XCCL_OK != xccl_context_create(lib, &ctx_params, cfg,
                                       &ctx->tls[tl_id].xccl_ctx)) {
        return XCCL_ERR_NO_MESSAGE;
    }
    xccl_context_config_release(cfg);

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
    int ctx_size = ctx->super.params.oob.size;
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

xccl_status_t xccl_hier_create_context(xccl_team_lib_t *lib,
                                       xccl_context_params_t *params,
                                       xccl_tl_context_config_t *config,
                                       xccl_tl_context_t **context)
{
    xccl_hier_context_t *ctx = (xccl_hier_context_t *)malloc(sizeof(*ctx));
    xccl_tl_hier_context_config_t *hier_cfg;
    int                 i;
    uint64_t            tl;

    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    ctx->procs = (xccl_hier_proc_data_t*)malloc(
        params->oob.size*sizeof(xccl_hier_proc_data_t));
    ctx->local_proc.socketid = xccl_local_process_info()->socketid;
    ctx->local_proc.node_hash = xccl_local_process_info()->node_hash;
    memset(ctx->tls, 0, sizeof(ctx->tls));
    *context = NULL;
    
    ucs_for_each_bit(tl, XCCL_TL_ALL) {
        ctx->tls[tl].enabled = 1;
    }

    hier_cfg = ucs_derived_of(config, xccl_tl_hier_context_config_t);
    /* Disable recursion */
    ctx->tls[ucs_ilog2(XCCL_TL_HIER)].enabled   = 0;
    ctx->tls[ucs_ilog2(XCCL_TL_MRAIL)].enabled  = 0;
    ctx->tls[ucs_ilog2(XCCL_TL_SHARP)].enabled  = hier_cfg->enable_sharp;
    ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled = hier_cfg->enable_shmseg;
    ctx->tls[ucs_ilog2(XCCL_TL_VMC)].enabled    = hier_cfg->enable_vmc;
    ctx->bcast_pipeline_thresh                  = hier_cfg->bcast_pipeline_thresh;
    ctx->bcast_pipeline_depth                   = hier_cfg->bcast_pipeline_depth;
    ctx->use_sm_get_bcast                       = hier_cfg->bcast_sm_get;
    ctx->bcast_sm_get_thresh                    = hier_cfg->bcast_sm_get_thresh;

    ucs_for_each_bit(tl, XCCL_TL_ALL) {
        if (XCCL_OK != xccl_hier_init_tl(ctx, tl, params->oob, config->env_prefix)) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }

    xccl_oob_allgather(&ctx->local_proc, ctx->procs,
                       sizeof(xccl_hier_proc_data_t), &params->oob);
    compute_layout(ctx);
    *context = &ctx->super;

    return XCCL_OK;
}

xccl_status_t xccl_hier_destroy_context(xccl_tl_context_t *team_context)
{
    xccl_hier_context_t    *ctx = ucs_derived_of(team_context, xccl_hier_context_t);
    xccl_oob_collectives_t *oob = &team_context->params.oob;
    int i;

    for (i=0; i<ucs_ilog2(XCCL_TL_LAST-1)+1; i++) {
        if (ctx->tls[i].enabled) {
            assert(ctx->tls[i].xccl_ctx);
            xccl_context_destroy(ctx->tls[i].xccl_ctx);
        }
    }
    free(ctx->procs);
    free(ctx);
    return XCCL_OK;
}

xccl_status_t xccl_hier_context_progress(xccl_tl_context_t *team_context)
{
    xccl_hier_context_t *ctx = ucs_derived_of(team_context, xccl_hier_context_t);
    int                 i;
    xccl_context_h      tl_ctx;

    for (i=0; i<ucs_ilog2(XCCL_TL_LAST-1)+1; i++) {
        if (ctx->tls[i].enabled) {
            tl_ctx = ctx->tls[i].xccl_ctx;
            assert(tl_ctx);
            xccl_context_progress(tl_ctx);
        }
    }

    return XCCL_OK;
}
