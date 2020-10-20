/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_CONTEXT_H_
#define XCCL_CONTEXT_H_

#include <api/xccl.h>
#include <xccl_lib.h>
#include <xccl_team_lib.h>
#include "topo/xccl_topo.h"

typedef struct xccl_context {
    xccl_lib_t             *lib;
    xccl_context_params_t  params;
    xccl_tl_context_t      **tl_ctx;
    int                    n_tl_ctx;
    xccl_topo_t            *topo;
} xccl_context_t;

typedef struct xccl_context_config {
    xccl_lib_t               *lib;
    xccl_tl_context_config_t **configs;
    int                      n_tl_cfg;
}xccl_context_config_t;

#endif
