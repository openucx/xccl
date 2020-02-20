/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TCCL_TEAM_SHARP_COLLECTIVE_H_
#define TCCL_TEAM_SHARP_COLLECTIVE_H_
#include "tccl_sharp_lib.h"

typedef struct tccl_sharp_coll_req {
    tccl_coll_req_t                 super;
    tccl_sharp_team_t              *team;
    struct sharp_coll_reduce_spec   reduce_spec;
    struct sharp_coll_comm         *sharp_comm;
    void                           *handle;
    tccl_sharp_buf_t               *sharp_buf;
    tccl_collective_type_t          coll_type;
    int                            (*start)(struct tccl_sharp_coll_req* req);
} tccl_sharp_coll_req_t;

tccl_status_t tccl_sharp_collective_init(tccl_coll_op_args_t *coll_args, tccl_coll_req_h *request,
                                         tccl_team_h team);
tccl_status_t tccl_sharp_collective_post(tccl_coll_req_h request);
tccl_status_t tccl_sharp_collective_wait(tccl_coll_req_h request);
tccl_status_t tccl_sharp_collective_test(tccl_coll_req_h request);
tccl_status_t tccl_sharp_collective_finalize(tccl_coll_req_h request);
#endif
