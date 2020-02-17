/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_TEAM_LIB_VMC_H_
#define TCCL_TEAM_LIB_VMC_H_
#include "tccl_team_lib.h"
#include <vmc.h>

typedef struct tccl_team_lib_vmc {
    tccl_team_lib_t super;
} tccl_team_lib_vmc_t;

typedef struct tccl_team_vmc_context {
    tccl_team_context_t super;
    vmc_ctx_h          vmc_ctx;
} tccl_team_vmc_context_t;

typedef struct tccl_team_vmc {
    tccl_team_t super;
    vmc_comm_h vmc_comm;
} tccl_team_vmc_t;

typedef struct tccl_team_vmc_coll_req {
    tccl_coll_req_t super;
    tccl_team_vmc_t *team;
    void           *handle;
    void           *buf;
    size_t         len;
    int            root;
} tccl_team_vmc_coll_req_t;

tccl_status_t tccl_team_vmc_collective_init(tccl_coll_op_args_t *coll_args,
                                          tccl_coll_req_h *request,
                                          tccl_team_h team);
tccl_status_t tccl_team_vmc_collective_post(tccl_coll_req_h request);
tccl_status_t tccl_team_vmc_collective_wait(tccl_coll_req_h request);
tccl_status_t tccl_team_vmc_collective_test(tccl_coll_req_h request);
tccl_status_t tccl_team_vmc_collective_finalize(tccl_coll_req_h request);
#endif
