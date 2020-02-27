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
extern tccl_team_lib_vmc_t tccl_team_lib_vmc;

typedef struct tccl_vmc_context {
    tccl_tl_context_t     super;
    vmc_ctx_h             vmc_ctx;
} tccl_vmc_context_t;

typedef struct tccl_vmc_team {
    tccl_team_t super;
    vmc_comm_h  vmc_comm;
} tccl_vmc_team_t;

typedef struct tccl_vmc_coll_req {
    tccl_coll_req_t  super;
    tccl_vmc_team_t *team;
    void            *handle;
    void            *buf;
    size_t           len;
    int              root;
} tccl_vmc_coll_req_t;

#endif
