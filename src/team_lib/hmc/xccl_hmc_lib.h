/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TEAM_LIB_HMC_H_
#define XCCL_TEAM_LIB_HMC_H_
#include "xccl_team_lib.h"
#include <hmc.h>

typedef struct xccl_tl_hmc_context_config {
    xccl_tl_context_config_t super;
    ucs_config_names_array_t devices;
} xccl_tl_hmc_context_config_t;

typedef struct xccl_team_lib_hmc {
    xccl_team_lib_t super;
} xccl_team_lib_hmc_t;
extern xccl_team_lib_hmc_t xccl_team_lib_hmc;

typedef struct xccl_hmc_context {
    xccl_tl_context_t     super;
    hmc_ctx_h             hmc_ctx;
} xccl_hmc_context_t;

typedef struct xccl_hmc_team {
    xccl_tl_team_t super;
    hmc_comm_h  hmc_comm;
} xccl_hmc_team_t;

typedef struct xccl_hmc_coll_req {
    xccl_tl_coll_req_t super;
    xccl_hmc_team_t    *team;
    void               *handle;
    void               *buf;
    size_t             len;
    int                root;
} xccl_hmc_coll_req_t;

#endif
