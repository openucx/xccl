/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_TEAM_H_
#define XCCL_TEAM_H_

#include <api/xccl.h>
#include <xccl_context.h>
#include <xccl_team_lib.h>
#include <ucs/memory/memory_type.h>

#define XCCL_CHECK_TEAM(_team)                                                    \
    do {                                                                          \
        if (_team->status != XCCL_OK) {                                           \
            xccl_error("team %p is used before team_create is completed", _team); \
            return XCCL_ERR_INVALID_PARAM;                                        \
        }                                                                         \
    } while(0)

typedef struct xccl_team {
    xccl_context_t     *ctx;
    int                coll_team_id[XCCL_COLL_LAST][UCS_MEMORY_TYPE_LAST];
    int                n_teams;
    int                last_team_create_posted;
    xccl_status_t      status;
    xccl_team_params_t params;
    xccl_tl_team_t     *tl_teams[1];
} xccl_team_t;

#endif
