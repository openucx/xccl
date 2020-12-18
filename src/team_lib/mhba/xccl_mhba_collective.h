/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_COLLECTIVE_H_
#define XCCL_MHBA_COLLECTIVE_H_

#include <xccl_mhba_lib.h>
#include <infiniband/verbs.h>
#include <ucs/arch/bitops.h>

#define squared(num) (num*num)

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
    int                           seq_num;
    struct ibv_mr                 *send_bf_mr;
    struct ibv_mr                 *receive_bf_mr;
    xccl_tl_coll_req_t            *barrier_req;
    int                           block_size;
    int                           started;
    struct ibv_mr                 *transpose_buf_mr;
    void                          *tmp_transpose_buf;
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
