/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_TEAM_LIB_SHARP_H_
#define TCCL_TEAM_LIB_SHARP_H_
#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>
#include "tccl_team_lib.h"

#define TCCL_TEAM_SHARP_REG_BUF_SIZE 1024
#define TCCL_TEAM_SHARP_REG_BUF_NUM  10

typedef struct tccl_team_lib_sharp {
    tccl_team_lib_t super;
} tccl_team_lib_sharp_t;

typedef struct tccl_team_sharp_context {
    tccl_team_context_t        super;
    tccl_oob_collectives_t     oob;
    struct sharp_coll_context *sharp_context;
} tccl_team_sharp_context_t;

typedef struct tccl_team_sharp_buf {
    void *buf;
    void *mr;
    void *orig_src_buf;
    void *orig_dst_buf;
    int  used;
} tccl_team_sharp_buf_t;

typedef struct tccl_team_sharp {
    tccl_team_t             super;
    tccl_oob_collectives_t  oob;
    struct sharp_coll_comm *sharp_comm;
    tccl_team_sharp_buf_t   bufs[TCCL_TEAM_SHARP_REG_BUF_NUM];
} tccl_team_sharp_t;

tccl_status_t tccl_team_sharp_collective_init(tccl_coll_op_args_t *coll_args, tccl_coll_req_h *request,
                                            tccl_team_h team);
tccl_status_t tccl_team_sharp_collective_post(tccl_coll_req_h request);
tccl_status_t tccl_team_sharp_collective_wait(tccl_coll_req_h request);
tccl_status_t tccl_team_sharp_collective_test(tccl_coll_req_h request);
tccl_status_t tccl_team_sharp_collective_finalize(tccl_coll_req_h request);

#endif
