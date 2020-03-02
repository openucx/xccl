/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_hier_lib.h"
#include "xccl_hier_team.h"
#include "xccl_hier_context.h"
#include "xccl_hier_schedule.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>


static inline xccl_status_t
xccl_hier_allreduce_init(xccl_coll_op_args_t *coll_args,
                         xccl_coll_req_h *request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    coll_schedule_t *schedule;
    xccl_hier_context_t *ctx = xccl_derived_of(team->ctx, xccl_hier_context_t);
    int top_lvl_pair = XCCL_HIER_PAIR_NODE_LEADERS_UCX;
    int sock_pair = XCCL_HIER_PAIR_SOCKET_SHMSEG;
    int sock_lead_pair = XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG;

    if (ctx->tls[XCCL_TL_SHARP].enabled) {
        top_lvl_pair = XCCL_HIER_PAIR_NODE_LEADERS_SHARP;
    }
    if (!ctx->tls[XCCL_TL_SHMSEG].enabled) {
        sock_pair = XCCL_HIER_PAIR_SOCKET_UCX;
        sock_lead_pair = XCCL_HIER_PAIR_SOCKET_LEADERS_UCX;
    }
    build_allreduce_schedule_3lvl(xccl_derived_of(team, xccl_hier_team_t),
                                  &schedule, (*coll_args),
                                  sock_pair, sock_lead_pair, top_lvl_pair);
    schedule->super.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->super;
    return XCCL_OK;
}


static inline xccl_status_t
xccl_hier_bcast_init(xccl_coll_op_args_t *coll_args,
                     xccl_coll_req_h *request, xccl_tl_team_t *team)
{
    coll_schedule_t *schedule;
    xccl_hier_context_t *ctx = xccl_derived_of(team->ctx, xccl_hier_context_t);
    int top_lvl_pair = XCCL_HIER_PAIR_NODE_LEADERS_UCX;

    if (ctx->tls[XCCL_TL_VMC].enabled) {
        top_lvl_pair = XCCL_HIER_PAIR_NODE_LEADERS_VMC;
    }
    build_bcast_schedule_3lvl(xccl_derived_of(team, xccl_hier_team_t),
                              &schedule, (*coll_args), top_lvl_pair);

    schedule->super.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->super;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_hier_barrier_init(xccl_coll_op_args_t *coll_args,
                      xccl_coll_req_h *request, xccl_tl_team_t *team)
{
    coll_schedule_t *schedule;
    xccl_hier_context_t *ctx = xccl_derived_of(team->ctx, xccl_hier_context_t);
    int top_lvl_pair = XCCL_HIER_PAIR_NODE_LEADERS_UCX;
    int sock_pair = XCCL_HIER_PAIR_SOCKET_SHMSEG;
    int sock_lead_pair = XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG;

    if (ctx->tls[XCCL_TL_SHARP].enabled) {
        top_lvl_pair = XCCL_HIER_PAIR_NODE_LEADERS_SHARP;
    }
    if (!ctx->tls[XCCL_TL_SHMSEG].enabled) {
        sock_pair = XCCL_HIER_PAIR_SOCKET_UCX;
        sock_lead_pair = XCCL_HIER_PAIR_SOCKET_LEADERS_UCX;
    }
    build_barrier_schedule_3lvl(xccl_derived_of(team, xccl_hier_team_t), &schedule,
                                sock_pair, sock_lead_pair, top_lvl_pair);
    schedule->super.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->super;
    return XCCL_OK;
}

static xccl_status_t
xccl_hier_collective_init(xccl_coll_op_args_t *coll_args,
                         xccl_coll_req_h *request, xccl_tl_team_t *team)
{
    switch (coll_args->coll_type) {
    case XCCL_ALLREDUCE:
        return xccl_hier_allreduce_init(coll_args, request, team);
    case XCCL_BARRIER:
        return xccl_hier_barrier_init(coll_args, request, team);
    case XCCL_BCAST:
        return xccl_hier_bcast_init(coll_args, request, team);
    }
    return XCCL_ERR_INVALID_PARAM;
}

static xccl_status_t xccl_hier_collective_post(xccl_coll_req_h request)
{
    coll_schedule_t *schedule = xccl_derived_of(request, coll_schedule_t);
    return coll_schedule_progress(schedule);
}

static xccl_status_t xccl_hier_collective_test(xccl_coll_req_h request)
{
    coll_schedule_t *schedule = xccl_derived_of(request, coll_schedule_t);
    coll_schedule_progress(schedule);
    return schedule->n_completed_colls == schedule->n_colls ?
        XCCL_OK : XCCL_INPROGRESS;
}

static xccl_status_t xccl_hier_collective_wait(xccl_coll_req_h request)
{
    xccl_status_t status = xccl_hier_collective_test(request);
    while (XCCL_OK != status) {
        status = xccl_hier_collective_test(request);
    }
    return XCCL_OK;
}

static xccl_status_t xccl_hier_collective_finalize(xccl_coll_req_h request)
{
    free(request);
    return XCCL_OK;
}

xccl_team_lib_hier_t xccl_team_lib_hier = {
    .super.name                 = "hier",
    .super.priority             = 150,
    .super.params.reproducible  = XCCL_LIB_NON_REPRODUCIBLE,
    .super.params.thread_mode   = XCCL_LIB_THREAD_SINGLE | XCCL_LIB_THREAD_MULTIPLE,
    .super.params.team_usage    = XCCL_USAGE_SW_COLLECTIVES,
    .super.params.coll_types    = XCCL_COLL_CAP_BARRIER |
                                  XCCL_COLL_CAP_BCAST | XCCL_COLL_CAP_ALLREDUCE,
    .super.ctx_create_mode      = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.create_team_context  = xccl_hier_create_context,
    .super.destroy_team_context = xccl_hier_destroy_context,
    .super.team_create_post     = xccl_hier_team_create_post,
    .super.team_destroy         = xccl_hier_team_destroy,
    .super.progress             = xccl_hier_context_progress,
    .super.collective_init      = xccl_hier_collective_init,
    .super.collective_post      = xccl_hier_collective_post,
    .super.collective_wait      = xccl_hier_collective_wait,
    .super.collective_test      = xccl_hier_collective_test,
    .super.collective_finalize  = xccl_hier_collective_finalize
};
