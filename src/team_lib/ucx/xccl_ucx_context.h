/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_UCX_CONTEXT_H_
#define XCCL_UCX_CONTEXT_H_
#include "xccl_ucx_lib.h"

typedef struct xccl_team_lib_ucx_context {
    xccl_tl_context_t     super;
    ucp_ep_h              *ucp_eps;
    ucp_address_t         *worker_address;
    int                   ucx_inited;
    ucp_context_h         ucp_context;
    ucp_worker_h          ucp_worker;
    size_t                ucp_addrlen;
    int                   num_to_probe;
    int                   next_cid;
} xccl_team_lib_ucx_context_t;

xccl_status_t xccl_ucx_create_context(xccl_team_lib_t *lib, xccl_context_config_t *config,
                                      xccl_tl_context_t **context);
xccl_status_t xccl_ucx_destroy_context(xccl_tl_context_t *context);

#endif
