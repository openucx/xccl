/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_TEAM_LIB_HIER_H_
#define TCCL_TEAM_LIB_HIER_H_
#include "tccl_team_lib.h"

typedef struct tccl_team_lib_hier {
    tccl_team_lib_t super;
} tccl_team_lib_hier_t;
extern tccl_team_lib_hier_t tccl_team_lib_hier;

typedef struct tccl_hier_collreq {
    tccl_coll_req_t     super;
    tccl_coll_op_args_t args;
    tccl_tl_team_t     *team;
    tccl_status_t       complete;
    tccl_status_t       (*start)(struct tccl_hier_collreq* req);
    tccl_status_t       (*progress)(struct tccl_hier_collreq* req);
} tccl_hier_collreq_t;

#endif
