/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TEAM_LIB_HIER_H_
#define XCCL_TEAM_LIB_HIER_H_
#include "xccl_team_lib.h"

typedef struct xccl_tl_hier_context_config {
    xccl_tl_context_config_t super;
    ucs_config_names_array_t devices;
    int                      enable_sharp;
    int                      enable_shmseg;
    int                      enable_vmc;
    size_t                   bcast_pipeline_thresh;
    unsigned                 bcast_pipeline_depth;
    int                      bcast_sm_get;
    size_t                   bcast_sm_get_thresh;
} xccl_tl_hier_context_config_t;

typedef struct xccl_team_lib_hier {
    xccl_team_lib_t super;
} xccl_team_lib_hier_t;
extern xccl_team_lib_hier_t xccl_team_lib_hier;

#endif
