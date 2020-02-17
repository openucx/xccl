/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_UCX_CONTEXT_H_
#define TCCL_UCX_CONTEXT_H_
#include "tccl_ucx_lib.h"

typedef struct tccl_team_lib_ucx_context {
    tccl_team_context_t    super;
    ucp_ep_h              *ucp_eps;
    ucp_address_t         *worker_address;
    int                   ucx_inited;
    ucp_context_h         ucp_context;
    ucp_worker_h          ucp_worker;
    size_t                ucp_addrlen;
    int                   num_to_probe;
    int                   next_cid;
} tccl_team_lib_ucx_context_t;

tccl_status_t tccl_ucx_create_context(tccl_team_lib_t *lib, tccl_team_context_config_t *config,
                                    tccl_team_context_t **context);
tccl_status_t tccl_ucx_destroy_context(tccl_team_context_t *context);

#endif
