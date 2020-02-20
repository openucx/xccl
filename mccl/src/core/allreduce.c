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

mccl_status_t mccl_allreduce_init(void *sbuf, void*rbuf, int count, tccl_dt_t dtype,
                                  tccl_op_t op, mccl_comm_h comm, mccl_request_h *req)
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

    build_allreduce_schedule_3lvl(comm, &schedule, MCCL_COLL_SCHED_SEQ, count, dtype, op, sbuf, rbuf,
                                  sock_team, sock_lead_team, top_lvl_team);

    *req = (mccl_request_h)schedule;
    return MCCL_SUCCESS;
}

mccl_status_t mccl_allreduce(void *sbuf, void*rbuf, int count, tccl_dt_t dtype,
                  tccl_op_t op, mccl_comm_h comm)
{
    mccl_request_h req;
    mccl_allreduce_init(sbuf, rbuf, count, dtype, op, comm, &req);
    mccl_start(req);
    mccl_wait(req);
    mccl_request_free(req);
    return MCCL_SUCCESS;
}
