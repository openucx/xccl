/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_HIER_CONTEXT_H_
#define TCCL_HIER_CONTEXT_H_
#include "tccl_hier_lib.h"
#include "api/tccl_tls.h"
typedef struct tccl_hier_proc_data {
    unsigned long node_hash;
    int node_id;
    int socketid; //if process is bound to a socket
} tccl_hier_proc_data_t;

typedef struct tccl_hier_tl_t {
    tccl_context_h tccl_ctx;
    int            enabled;
} tccl_hier_tl_t;

typedef struct tccl_hier_context {
    tccl_tl_context_t         super;
    tccl_hier_proc_data_t     local_proc; // local proc data
    tccl_hier_proc_data_t    *procs; // data for all processes
    tccl_hier_tl_t            tls[TCCL_TL_LAST];
    int                       nnodes;
} tccl_hier_context_t;

tccl_status_t tccl_hier_create_context(tccl_team_lib_t *lib, tccl_context_config_t *config,
                                       tccl_tl_context_t **context);
tccl_status_t tccl_hier_destroy_context(tccl_tl_context_t *context);

#endif
