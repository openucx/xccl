/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_UCX_CONTEXT_H_
#define XCCL_UCX_CONTEXT_H_
#include "xccl_ucx_lib.h"

typedef struct xccl_team_lib_ucx_context {
    xccl_tl_context_t super;
    ucp_ep_h          *ucp_eps;
    ucp_address_t     *worker_address;
    int               ucx_inited;
    ucp_context_h     ucp_context;
    ucp_worker_h      ucp_worker;
    size_t            ucp_addrlen;
    int               next_cid;
    unsigned          num_to_probe;
    unsigned          barrier_kn_radix;
    unsigned          bcast_kn_radix;
    unsigned          allreduce_kn_radix;
    unsigned          allreduce_alg_id;
    unsigned          reduce_kn_radix;
    unsigned          alltoall_pairwise_chunk;
    int               alltoall_pairwise_reverse;
    unsigned          alltoall_pairwise_barrier;
    int               block_stream;
    unsigned          pre_mem_map;
} xccl_team_lib_ucx_context_t;

xccl_status_t xccl_ucx_create_context(xccl_team_lib_t *lib,
                                      xccl_context_params_t *params,
                                      xccl_tl_context_config_t *config,
                                      xccl_tl_context_t **context);
xccl_status_t xccl_ucx_destroy_context(xccl_tl_context_t *context);

void xccl_ucx_mem_map(void *addr, size_t length, ucs_memory_type_t mem_type,
                      xccl_team_lib_ucx_context_t *ctx);
#define TEAM_UCX_CTX(_team) (ucs_derived_of((_team)->super.ctx, xccl_team_lib_ucx_context_t))
#define TEAM_UCX_CTX_REQ(_req) (ucs_derived_of((_req)->team->ctx, xccl_team_lib_ucx_context_t))
#define TEAM_UCX_WORKER(_team) TEAM_UCX_CTX(_team)->ucp_worker

enum {
    XCCL_TEAM_UCX_NO_PRE_MAP                    = 0,
    XCCL_TEAM_UCX_COLL_INIT_PRE_MAP             = 1,
    XCCL_TEAM_UCX_ALLOC_PRE_MAP                 = 2,
    XCCL_TEAM_UCX_COLL_INIT_AND_ALLOC_PRE_MAP   = 3,
};
#endif
