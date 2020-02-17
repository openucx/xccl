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

int mccl_bcast_init(void *buf,int count, tccl_dt_t dtype,
                   int root, mccl_comm_h comm, mccl_request_h *req)
{
    coll_schedule_t *schedule;
    mccl_comm_t *mccl_comm = (mccl_comm_t*)comm;
    mccl_context_t *ctx = mccl_comm->config.mccl_ctx;
    int top_lvl_team = MCCL_TEAM_NODE_LEADERS_UCX;

    if (ctx->libs[TCCL_LIB_VMC].enabled) {
        top_lvl_team = MCCL_TEAM_NODE_LEADERS_VMC;
    }
    build_bcast_schedule_3lvl(comm, &schedule, buf, count, dtype,
                              root, top_lvl_team);

    *req = (mccl_request_h)schedule;
    return TCCL_OK;
}

int mccl_bcast(void* buf, int count, tccl_dt_t dtype,
              int root, mccl_comm_h comm)
{
    mccl_request_h req;
    mccl_bcast_init(buf, count, dtype, root, comm, &req);
    mccl_start(req);
    mccl_wait(req);
    /* fprintf(stderr, "Bcast done\n"); */
    mccl_request_free(req);
    return TCCL_OK;
}
