/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_COLLECTIVE_H_
#define XCCL_MHBA_COLLECTIVE_H_

#include <xccl_mhba_lib.h>
#include <ucs/arch/bitops.h>

typedef struct xccl_mhba_coll_req xccl_mhba_coll_req_t;
typedef struct xccl_mhba_task {
    xccl_coll_task_t      super;
    xccl_mhba_coll_req_t *req;
} xccl_mhba_task_t;

typedef struct xccl_mhba_coll_req {
    xccl_tl_coll_req_t  super;
    xccl_schedule_t     schedule;
    xccl_mhba_task_t   *tasks;
    xccl_coll_op_args_t args;
    xccl_mhba_team_t   *team;
    uint64_t            seq_num;
    int                 need_update_send_mkey;
    int                 need_update_recv_mkey;
    int                 send_buffer_reg_change_flag;
    int                 recv_buffer_reg_change_flag;
    xccl_tl_coll_req_t *barrier_req;
    int                 block_size;
    int                 started;
    xccl_mhba_reg_t    *send_rcache_region_p;
    xccl_mhba_reg_t    *recv_rcache_region_p;
    struct ibv_mr      *transpose_buf_mr;
    void               *tmp_transpose_buf;
} xccl_mhba_coll_req_t;

xccl_status_t xccl_mhba_collective_init_base(xccl_coll_op_args_t   *coll_args,
                                             xccl_mhba_coll_req_t **request,
                                             xccl_mhba_team_t      *team);

xccl_status_t xccl_mhba_alltoall_init(xccl_coll_op_args_t  *coll_args,
                                      xccl_mhba_coll_req_t *request,
                                      xccl_mhba_team_t     *team);

#endif
