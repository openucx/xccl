/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "mccl_core.h"
#include "schedule.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int mccl_barrier_init(mccl_comm_h comm, mccl_request_h *req)
{
    coll_schedule_t *schedule;
    mccl_comm_t *mccl_comm = (mccl_comm_t*)comm;
    mccl_context_t *ctx = mccl_comm->config.mccl_ctx;
    int top_lvl_team = MCCL_TEAM_NODE_LEADERS_UCX;
    int sock_team = MCCL_TEAM_SOCKET_SHMSEG;
    int sock_lead_team = MCCL_TEAM_SOCKET_LEADERS_SHMSEG;

    if (ctx->libs[TCCL_LIB_SHARP].enabled) {
        top_lvl_team = MCCL_TEAM_NODE_LEADERS_SHARP;
    }

    if (!ctx->libs[TCCL_LIB_SHMSEG].enabled) {
        sock_team = MCCL_TEAM_SOCKET_UCX;
        sock_lead_team = MCCL_TEAM_SOCKET_LEADERS_UCX;
    }

    build_barrier_schedule_3lvl(comm, &schedule, MCCL_COLL_SCHED_SEQ,
                                sock_team, sock_lead_team, top_lvl_team);

    *req = (mccl_request_h)schedule;
    return TCCL_OK;
}
