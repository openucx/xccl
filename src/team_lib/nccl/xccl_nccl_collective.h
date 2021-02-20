/*
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_NCCL_COLLECTIVE_H_
#define XCCL_NCCL_COLLECTIVE_H_

#include <xccl_nccl_lib.h>
#include <utils/mem_component.h>
#include <nccl.h>

#define ncclOpUnsupported (ncclNumOps + 1)
#define ncclDataTypeUnsupported (ncclNumTypes + 1)

typedef xccl_status_t (*xccl_nccl_collective_start_fn)(xccl_tl_coll_req_t *req);

typedef struct xccl_nccl_coll_req {
    xccl_tl_coll_req_t            super;
    xccl_coll_op_args_t           args;
    xccl_nccl_team_t              *team;
    xccl_nccl_collective_start_fn coll_start;
    xccl_mc_event_t               *completed;
    xccl_status_t                 status;
} xccl_nccl_coll_req_t;

xccl_status_t
xccl_nccl_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_nccl_coll_req_t **request,
                               xccl_nccl_team_t *team);

xccl_status_t
xccl_nccl_allreduce_init(xccl_coll_op_args_t *coll_args,
                         xccl_nccl_coll_req_t *request,
                         xccl_nccl_team_t *team);

xccl_status_t
xccl_nccl_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_nccl_coll_req_t *request,
                        xccl_nccl_team_t *team);

xccl_status_t
xccl_nccl_alltoallv_init(xccl_coll_op_args_t *coll_args,
                         xccl_nccl_coll_req_t *request,
                         xccl_nccl_team_t *team);

xccl_status_t
xccl_nccl_allgather_init(xccl_coll_op_args_t *coll_args,
                         xccl_nccl_coll_req_t *request,
                         xccl_nccl_team_t *team);
xccl_status_t
xccl_nccl_bcast_init(xccl_coll_op_args_t *coll_args,
                     xccl_nccl_coll_req_t *request,
                     xccl_nccl_team_t *team);

#define TEAM_NCCL_CTX_REQ(_req) \
        (ucs_derived_of((_req)->team->super.ctx, xccl_nccl_context_t))

#endif
