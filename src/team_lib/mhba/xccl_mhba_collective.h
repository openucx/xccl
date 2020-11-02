/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_COLLECTIVE_H_
#define XCCL_MHBA_COLLECTIVE_H_

#include <xccl_mhba_lib.h>
#include <infiniband/verbs.h>

typedef struct xccl_mhba_coll_req xccl_mhba_coll_req_t;
typedef struct xccl_mhba_task {
    xccl_coll_task_t super;
    xccl_mhba_coll_req_t *req;
} xccl_mhba_task_t;

typedef struct xccl_mhba_coll_req {
    xccl_tl_coll_req_t            super;
    xccl_schedule_t               schedule;
    xccl_mhba_task_t              *tasks;
    xccl_coll_op_args_t           args;
    xccl_mhba_team_t              *team;
    int                           asr_rank;
    int                           seq_num;
    xccl_tl_coll_req_t            *barrier_req;
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
