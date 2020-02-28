/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "tccl_hier_lib.h"
#include "tccl_hier_team.h"
#include "tccl_hier_context.h"
#include "tccl_hier_schedule.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>


static inline tccl_status_t
tccl_hier_allreduce_init(tccl_coll_op_args_t *coll_args,
                         tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    coll_schedule_t *schedule;
    tccl_hier_context_t *ctx = tccl_derived_of(team->ctx, tccl_hier_context_t);
    int top_lvl_pair = TCCL_HIER_PAIR_NODE_LEADERS_UCX;
    int sock_pair = TCCL_HIER_PAIR_SOCKET_SHMSEG;
    int sock_lead_pair = TCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG;

    if (ctx->tls[TCCL_TL_SHARP].enabled) {
        top_lvl_pair = TCCL_HIER_PAIR_NODE_LEADERS_SHARP;
    }
    if (!ctx->tls[TCCL_TL_SHMSEG].enabled) {
        sock_pair = TCCL_HIER_PAIR_SOCKET_UCX;
        sock_lead_pair = TCCL_HIER_PAIR_SOCKET_LEADERS_UCX;
    }
    build_allreduce_schedule_3lvl(tccl_derived_of(team, tccl_hier_team_t),
                                  &schedule, (*coll_args),
                                  sock_pair, sock_lead_pair, top_lvl_pair);
    schedule->super.lib = &tccl_team_lib_hier.super;
    (*request) = &schedule->super;
    return TCCL_OK;
}


static inline tccl_status_t
tccl_hier_bcast_init(tccl_coll_op_args_t *coll_args,
                     tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    coll_schedule_t *schedule;
    tccl_hier_context_t *ctx = tccl_derived_of(team->ctx, tccl_hier_context_t);
    int top_lvl_pair = TCCL_HIER_PAIR_NODE_LEADERS_UCX;

    if (ctx->tls[TCCL_TL_VMC].enabled) {
        top_lvl_pair = TCCL_HIER_PAIR_NODE_LEADERS_VMC;
    }
    build_bcast_schedule_3lvl(tccl_derived_of(team, tccl_hier_team_t),
                              &schedule, (*coll_args), top_lvl_pair);

    schedule->super.lib = &tccl_team_lib_hier.super;
    (*request) = &schedule->super;
    return TCCL_OK;
}

static inline tccl_status_t
tccl_hier_barrier_init(tccl_coll_op_args_t *coll_args,
                      tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    coll_schedule_t *schedule;
    tccl_hier_context_t *ctx = tccl_derived_of(team->ctx, tccl_hier_context_t);
    int top_lvl_pair = TCCL_HIER_PAIR_NODE_LEADERS_UCX;
    int sock_pair = TCCL_HIER_PAIR_SOCKET_SHMSEG;
    int sock_lead_pair = TCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG;

    if (ctx->tls[TCCL_TL_SHARP].enabled) {
        top_lvl_pair = TCCL_HIER_PAIR_NODE_LEADERS_SHARP;
    }
    if (!ctx->tls[TCCL_TL_SHMSEG].enabled) {
        sock_pair = TCCL_HIER_PAIR_SOCKET_UCX;
        sock_lead_pair = TCCL_HIER_PAIR_SOCKET_LEADERS_UCX;
    }
    build_barrier_schedule_3lvl(tccl_derived_of(team, tccl_hier_team_t), &schedule,
                                sock_pair, sock_lead_pair, top_lvl_pair);
    schedule->super.lib = &tccl_team_lib_hier.super;
    (*request) = &schedule->super;
    return TCCL_OK;
}

static tccl_status_t
tccl_hier_collective_init(tccl_coll_op_args_t *coll_args,
                         tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    switch (coll_args->coll_type) {
    case TCCL_ALLREDUCE:
        return tccl_hier_allreduce_init(coll_args, request, team);
    case TCCL_BARRIER:
        return tccl_hier_barrier_init(coll_args, request, team);
    case TCCL_BCAST:
        return tccl_hier_bcast_init(coll_args, request, team);
    }
    return TCCL_ERR_INVALID_PARAM;
}

static tccl_status_t tccl_hier_collective_post(tccl_coll_req_h request)
{
    coll_schedule_t *schedule = tccl_derived_of(request, coll_schedule_t);
    return coll_schedule_progress(schedule);
}

static tccl_status_t tccl_hier_collective_test(tccl_coll_req_h request)
{
    coll_schedule_t *schedule = tccl_derived_of(request, coll_schedule_t);
    coll_schedule_progress(schedule);
    return schedule->n_completed_colls == schedule->n_colls ?
        TCCL_OK : TCCL_INPROGRESS;
}

static tccl_status_t tccl_hier_collective_wait(tccl_coll_req_h request)
{
    tccl_status_t status = tccl_hier_collective_test(request);
    while (TCCL_OK != status) {
        status = tccl_hier_collective_test(request);
    }
    return TCCL_OK;
}

static tccl_status_t tccl_hier_collective_finalize(tccl_coll_req_h request)
{
    free(request);
    return TCCL_OK;
}

tccl_team_lib_hier_t tccl_team_lib_hier = {
    .super.name                 = "hier",
    .super.priority             = 150,
    .super.params.reproducible  = TCCL_LIB_NON_REPRODUCIBLE,
    .super.params.thread_mode   = TCCL_LIB_THREAD_SINGLE | TCCL_LIB_THREAD_MULTIPLE,
    .super.params.team_usage    = TCCL_USAGE_SW_COLLECTIVES,
    .super.params.coll_types    = TCCL_COLL_CAP_BARRIER |
                                  TCCL_COLL_CAP_BCAST | TCCL_COLL_CAP_ALLREDUCE,
    .super.ctx_create_mode      = TCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.create_team_context  = tccl_hier_create_context,
    .super.destroy_team_context = tccl_hier_destroy_context,
    .super.team_create_post     = tccl_hier_team_create_post,
    .super.team_destroy         = tccl_hier_team_destroy,
    .super.progress             = NULL,
    .super.collective_init      = tccl_hier_collective_init,
    .super.collective_post      = tccl_hier_collective_post,
    .super.collective_wait      = tccl_hier_collective_wait,
    .super.collective_test      = tccl_hier_collective_test,
    .super.collective_finalize  = tccl_hier_collective_finalize
};
