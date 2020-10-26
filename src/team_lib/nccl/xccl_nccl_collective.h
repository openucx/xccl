/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_NCCL_COLLECTIVE_H_
#define XCCL_NCCL_COLLECTIVE_H_

#include <xccl_nccl_lib.h>
#include <nccl.h>

#define ncclOpUnsupported (ncclNumOps + 1)
#define ncclDataTypeUnsupported (ncclNumTypes + 1)

typedef xccl_status_t (*xccl_nccl_collective_start_fn)(xccl_tl_coll_req_t *req);

typedef struct xccl_nccl_coll_req {
    xccl_tl_coll_req_t            super;
    xccl_coll_op_args_t           args;
    xccl_nccl_team_t              *team;
    xccl_nccl_collective_start_fn coll_start;
    cudaEvent_t                   completed;
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

#endif
