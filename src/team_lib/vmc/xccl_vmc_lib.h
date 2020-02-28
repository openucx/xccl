/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TEAM_LIB_VMC_H_
#define XCCL_TEAM_LIB_VMC_H_
#include "xccl_team_lib.h"
#include <vmc.h>

typedef struct xccl_team_lib_vmc {
    xccl_team_lib_t super;
} xccl_team_lib_vmc_t;
extern xccl_team_lib_vmc_t xccl_team_lib_vmc;

typedef struct xccl_vmc_context {
    xccl_tl_context_t     super;
    vmc_ctx_h             vmc_ctx;
} xccl_vmc_context_t;

typedef struct xccl_vmc_team {
    xccl_tl_team_t super;
    vmc_comm_h  vmc_comm;
} xccl_vmc_team_t;

typedef struct xccl_vmc_coll_req {
    xccl_coll_req_t  super;
    xccl_vmc_team_t *team;
    void            *handle;
    void            *buf;
    size_t           len;
    int              root;
} xccl_vmc_coll_req_t;

#endif
