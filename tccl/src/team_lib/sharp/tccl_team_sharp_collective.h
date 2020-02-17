/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TCCL_TEAM_SHARP_COLLECTIVE_H_
#define TCCL_TEAM_SHARP_COLLECTIVE_H_
#include "tccl_team_lib_sharp.h"

typedef struct tccl_team_sharp_coll_req {
    tccl_coll_req_t                super;
    tccl_team_sharp_t              *team;
    struct sharp_coll_reduce_spec reduce_spec;
    struct sharp_coll_comm        *sharp_comm;
    void                          *handle;
    tccl_team_sharp_buf_t          *sharp_buf;
    tccl_collective_type_t         coll_type;
    tccl_status_t                  (*start)(struct tccl_team_sharp_coll_req* req);
} tccl_team_sharp_coll_req_t;

#endif