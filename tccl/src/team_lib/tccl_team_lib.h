/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#ifndef TCCL_TEAM_LIB_H_
#define TCCL_TEAM_LIB_H_
#include "api/tccl.h"
#include <ucs/config/types.h>
#include <ucs/sys/compiler_def.h>
#include <assert.h>
#include <string.h>

typedef struct tccl_team_lib {
    char*                               name;
    int                                 priority;
    tccl_lib_config_t                   config;
    tccl_team_lib_context_create_mode_t ctx_create_mode;
    void*                               dl_handle;
    tccl_status_t                       (*create_team_context)(tccl_team_lib_h lib,
                                                              tccl_context_config_h config,
                                                              tccl_context_h *team_context);
    tccl_status_t                       (*destroy_team_context)(tccl_context_h team_context);
    tccl_status_t                       (*progress)(tccl_context_h team_context);
    tccl_status_t                       (*team_create_post)(tccl_context_h team_ctx,
                                                            tccl_team_config_h config,
                                                            tccl_oob_collectives_t oob, tccl_team_h *team);
    tccl_status_t                       (*team_destroy)(tccl_team_h team);
    tccl_status_t                       (*collective_init)(tccl_coll_op_args_t *coll_args,
                                                           tccl_coll_req_h *request, tccl_team_h team);
    tccl_status_t                       (*collective_post)(tccl_coll_req_h request);
    tccl_status_t                       (*collective_wait)(tccl_coll_req_h request);
    tccl_status_t                       (*collective_test)(tccl_coll_req_h request);
    tccl_status_t                       (*collective_finalize)(tccl_coll_req_h request);
} tccl_team_lib_t;

typedef struct tccl_lib {
    tccl_lib_config_t          config_requested;
    ucs_log_component_config_t log_config;
    int                        n_libs_opened;
    int                        libs_array_size;
    char                       *lib_path;
    tccl_team_lib_t            **libs;
} tccl_lib_t;

typedef struct tccl_context {
    tccl_team_lib_t       *lib;
    tccl_context_config_t cfg;
} tccl_context_t;

typedef struct tccl_team {
    tccl_context_t         *ctx;
    tccl_team_config_t     cfg;
    tccl_oob_collectives_t oob;
} tccl_team_t;

static inline int tccl_team_rank_to_world(tccl_team_config_t *cfg, int rank)
{
    int r;
    switch(cfg->range.type) {
    case TCCL_EP_RANGE_STRIDED:
        r = cfg->range.strided.start + rank*cfg->range.strided.stride;
        break;
    case TCCL_EP_RANGE_MAP:
        r = cfg->range.map.map[rank];
        break;
    case TCCL_EP_RANGE_CB:
        r = cfg->range.cb.cb(rank, cfg->range.cb.cb_ctx);
        break;
    default:
        assert(0);
    }
    return r;
}

#define TCCL_TEAM_SUPER_INIT(_team, _ctx, _config, _oob) do {            \
        (_team).ctx = (_ctx);                                          \
        (_team).oob = (_oob);                                          \
        memcpy(&((_team).cfg), (_config), sizeof(tccl_team_config_t));  \
    }while(0)

#define TCCL_CONTEXT_SUPER_INIT(_ctx, _lib, _config) do {                \
        (_ctx).lib = (_lib);                                            \
        memcpy(&((_ctx).cfg), (_config), sizeof(tccl_context_config_t)); \
    }while(0)

#endif
