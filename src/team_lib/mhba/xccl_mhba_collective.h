/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_COLLECTIVE_H_
#define XCCL_MHBA_COLLECTIVE_H_

#include <xccl_mhba_lib.h>

typedef xccl_status_t (*xccl_mhba_collective_start_fn)(xccl_tl_coll_req_t *req);

typedef struct xccl_mhba_coll_req {
    xccl_tl_coll_req_t            super;
    xccl_coll_op_args_t           args;
    xccl_mhba_team_t              *team;
    xccl_mhba_collective_start_fn coll_start;
} xccl_mhba_coll_req_t;

xccl_status_t
xccl_mhba_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_mhba_coll_req_t **request,
                               xccl_mhba_team_t *team);

xccl_status_t
xccl_mhba_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_mhba_coll_req_t *request,
                        xccl_mhba_team_t *team);

#endif
