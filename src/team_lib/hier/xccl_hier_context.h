/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_HIER_CONTEXT_H_
#define XCCL_HIER_CONTEXT_H_
#include "xccl_hier_lib.h"
#include "api/xccl_tls.h"
typedef struct xccl_hier_proc_data {
    unsigned long node_hash;
    int node_id;
    int socketid; //if process is bound to a socket
} xccl_hier_proc_data_t;

typedef struct xccl_hier_tl_t {
    xccl_context_h xccl_ctx;
    int            enabled;
} xccl_hier_tl_t;

typedef struct xccl_hier_context {
    xccl_tl_context_t         super;
    xccl_hier_proc_data_t     local_proc; // local proc data
    xccl_hier_proc_data_t    *procs; // data for all processes
    xccl_hier_tl_t            tls[XCCL_TL_LAST];
    int                       nnodes;
    int                       max_ppn;
    int                       min_ppn;
} xccl_hier_context_t;

xccl_status_t xccl_hier_create_context(xccl_team_lib_t *lib, xccl_context_config_t *config,
                                       xccl_tl_context_t **context);
xccl_status_t xccl_hier_destroy_context(xccl_tl_context_t *context);

#endif
