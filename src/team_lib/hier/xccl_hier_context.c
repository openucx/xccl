/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_hier_context.h"
#include "xccl_hier_sbgp.h"
#include "utils/utils.h"
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <ucs/sys/math.h>

static xccl_status_t
xccl_hier_init_tl(xccl_hier_context_t *ctx, int tl_idx,
                  xccl_oob_collectives_t oob, xccl_tl_hier_context_config_t *hier_cfg, unsigned thread_mode) {
    xccl_team_lib_hier_t  *hlib = ucs_derived_of(ctx->super.lib,
                                                 xccl_team_lib_hier_t);
    xccl_lib_h            lib   = hlib->tl_lib;
    xccl_tl_id_t          tl_id = UCS_BIT(tl_idx);
    xccl_context_config_t *cfg;
    xccl_status_t         status;
    char                  env_prefix[128];

    if (!ctx->tls[tl_idx].enabled) {
        return XCCL_OK;
    }

    xccl_context_params_t ctx_params = {
        .field_mask      = XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE |
                           XCCL_CONTEXT_PARAM_FIELD_TEAM_COMPLETION_TYPE |
                           XCCL_CONTEXT_PARAM_FIELD_OOB |
                           XCCL_CONTEXT_PARAM_FIELD_TLS,
        .thread_mode     = thread_mode,
        .completion_type = XCCL_TEAM_COMPLETION_TYPE_BLOCKING,
        .oob             = oob,
        .tls             = tl_id,
    };

    if (hier_cfg->super.env_prefix != NULL) {
        snprintf(env_prefix, sizeof(env_prefix), "%s_HIER_%s",
                 hier_cfg->super.env_prefix, xccl_tl_str(tl_id));
    }
    else {
        snprintf(env_prefix, sizeof(env_prefix), "HIER_%s", xccl_tl_str(tl_id));
    }
    xccl_context_config_read(lib, env_prefix, NULL, &cfg);
    if (hier_cfg->devices.count > 1 ||
        0 != strncmp(hier_cfg->devices.names[0], "all", 3)) {
        char *dev_str = xccl_names_array_to_str(&hier_cfg->devices);
        if (dev_str) {
            xccl_context_config_modify(&tl_id, cfg, "NET_DEVICES", dev_str);
            free(dev_str);
        }
    }
    status = xccl_context_create(lib, &ctx_params, cfg, &ctx->tls[tl_idx].xccl_ctx);
    if (status != XCCL_OK) {
        xccl_hier_warn("Failed to open %s context", xccl_tl_str(tl_id));
    }
    xccl_context_config_release(cfg);
    return status;
}

static void compute_layout(xccl_hier_context_t *ctx) {
    int ctx_size = ctx->super.params.oob.size;
    xccl_hier_proc_data_t *sorted = (xccl_hier_proc_data_t*)
        malloc(ctx_size*sizeof(xccl_hier_proc_data_t));
    memcpy(sorted, ctx->procs, ctx_size*sizeof(xccl_hier_proc_data_t));
    qsort(sorted, ctx_size, sizeof(xccl_hier_proc_data_t),
          xccl_hier_compare_proc_data);
    unsigned long current_hash = sorted[0].node_hash;
    int current_ppn = 1;
    int min_ppn = INT_MAX;
    int max_ppn = 0;
    int nnodes = 1;
    int max_sockid = 0;
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
        if (ctx->procs[j].socketid > max_sockid) {
            max_sockid = ctx->procs[j].socketid;
        }
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
    ctx->max_n_sockets = max_sockid+1;
}

xccl_status_t xccl_hier_create_context(xccl_team_lib_t *lib,
                                       xccl_context_params_t *params,
                                       xccl_tl_context_config_t *config,
                                       xccl_tl_context_t **context)
{
    xccl_team_lib_hier_t *hlib = ucs_derived_of(lib, xccl_team_lib_hier_t);
    xccl_hier_context_t  *ctx  = (xccl_hier_context_t *)malloc(sizeof(*ctx));
    xccl_tl_hier_context_config_t *hier_cfg;
    xccl_status_t       status;
    int                 i;
    uint64_t            tl;
    if (!ctx) {
        return XCCL_ERR_NO_MEMORY;
    }
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);

    if (NULL == hlib->tl_lib) {
        xccl_lib_params_t lib_params = {
            .field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE,
            .team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                          XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
        };
        if (XCCL_OK != (status = xccl_lib_init(&lib_params, NULL, &hlib->tl_lib))) {
            return status;
        }
    }

    ctx->procs = (xccl_hier_proc_data_t*)malloc(
        params->oob.size*sizeof(xccl_hier_proc_data_t));
    ctx->local_proc.socketid  = xccl_local_process_info()->socketid;
    ctx->local_proc.node_hash = xccl_local_process_info()->node_hash;
    ctx->local_proc.pid       = xccl_local_process_info()->pid;
    memset(ctx->tls, 0, sizeof(ctx->tls));
    *context = NULL;

    ucs_for_each_bit(tl, XCCL_TL_ALL) {
        ctx->tls[tl].enabled = 1;
    }

    hier_cfg = ucs_derived_of(config, xccl_tl_hier_context_config_t);
    /* Disable recursion */
    ctx->tls[ucs_ilog2(XCCL_TL_HIER)].enabled    = 0;
    ctx->tls[ucs_ilog2(XCCL_TL_MRAIL)].enabled   = 0;
    ctx->tls[ucs_ilog2(XCCL_TL_DPU)].enabled     = 0;
    ctx->tls[ucs_ilog2(XCCL_TL_SHARP)].enabled   = hier_cfg->enable_sharp;
    ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled  = hier_cfg->enable_shmseg;
    ctx->tls[ucs_ilog2(XCCL_TL_HMC)].enabled     = hier_cfg->enable_hmc;
    ctx->tls[ucs_ilog2(XCCL_TL_NCCL)].enabled    = hier_cfg->enable_nccl;
    ctx->bcast_pipeline_thresh                   = hier_cfg->bcast_pipeline_thresh;
    ctx->bcast_pipeline_depth                    = hier_cfg->bcast_pipeline_depth;
    ctx->use_sm_get_bcast                        = hier_cfg->bcast_sm_get;
    ctx->bcast_sm_get_thresh                     = hier_cfg->bcast_sm_get_thresh;
    ctx->node_leader_rank_id                     = hier_cfg->node_leader_rank_id;

    ucs_for_each_bit(tl, XCCL_TL_ALL) {
        if (XCCL_OK != (status = xccl_hier_init_tl(ctx, tl, params->oob, hier_cfg, params->thread_mode))) {
            return status;
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
            xccl_status_t status = xccl_context_destroy(ctx->tls[i].xccl_ctx);
            if (status != XCCL_OK){
                return status;
            }
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
