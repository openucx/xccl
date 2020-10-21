/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_HIER_CONTEXT_H_
#define XCCL_HIER_CONTEXT_H_

#include "xccl_hier_lib.h"
#include "api/xccl_tls.h"
#include <xccl_ucs.h>


typedef struct xccl_hier_tl_t {
    xccl_context_h xccl_ctx;
    int            enabled;
} xccl_hier_tl_t;

typedef struct xccl_hier_context {
    xccl_tl_context_t         super;
    xccl_hier_tl_t            tls[ucs_ilog2(XCCL_TL_LAST-1)+1];
    int                       nnodes;
    int                       max_ppn;
    int                       min_ppn;
    int                       max_n_sockets;
    int                       node_leader_rank_id;
    int                       use_sm_get_bcast;
    int                       bcast_pipeline_depth;
    size_t                    bcast_sm_get_thresh;
    size_t                    bcast_pipeline_thresh;
} xccl_hier_context_t;

xccl_status_t xccl_hier_create_context(xccl_team_lib_t *lib,
                                       xccl_context_params_t *params,
                                       xccl_tl_context_config_t *config,
                                       xccl_tl_context_t **context);
xccl_status_t xccl_hier_destroy_context(xccl_tl_context_t *context);
xccl_status_t xccl_hier_context_progress(xccl_tl_context_t *team_ctx);
#endif
