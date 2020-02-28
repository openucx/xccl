/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "tccl_hier_lib.h"
/* #include "tccl_hier_team.h" */
#include "tccl_hier_context.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static inline tccl_status_t
tccl_hier_coll_base_init(tccl_coll_op_args_t *coll_args, tccl_tl_team_t *team,
                        tccl_hier_collreq_t **request)
{
    //todo malloc ->mpool
    tccl_hier_collreq_t *req = (tccl_hier_collreq_t *)malloc(sizeof(*req));
    memcpy(&req->args, coll_args, sizeof(*coll_args));
    req->complete  = TCCL_INPROGRESS;
    req->team      = team;
    req->super.lib = &tccl_team_lib_hier.super;
    (*request)     = req;
    return TCCL_OK;
}

static inline tccl_status_t
tccl_hier_allreduce_init(tccl_coll_op_args_t *coll_args,
                         tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    tccl_hier_collreq_t *req;
    tccl_hier_coll_base_init(coll_args, team, &req);
    /* req->start = tccl_hier_allreduce_knomial_start; */
    (*request) = (tccl_coll_req_h)req;
    return TCCL_OK;
}

static inline tccl_status_t
tccl_hier_bcast_init(tccl_coll_op_args_t *coll_args,
                     tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    tccl_hier_collreq_t *req;
    tccl_status_t status = TCCL_OK;
    tccl_hier_coll_base_init(coll_args, team, &req);
    if (!coll_args->alg.set_by_user) {
        /* Automatic algorithm selection - take knomial */
        /* req->start = tccl_hier_bcast_knomial_start; */
    } else {
        switch (coll_args->alg.id) {
        case 0:
            /* req->start = tccl_hier_bcast_linear_start; */
            break;
        case 1:
            /* req->start = tccl_hier_bcast_knomial_start; */
            break;
        default:
            free(req);
            req = NULL;
            status = TCCL_ERR_INVALID_PARAM;
        }
    }
    (*request) = (tccl_coll_req_h)req;
    return status;
}

static inline tccl_status_t
tccl_hier_barrier_init(tccl_coll_op_args_t *coll_args,
                      tccl_coll_req_h *request, tccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    tccl_hier_collreq_t *req;
    tccl_hier_coll_base_init(coll_args, team, &req);
    /* req->start = tccl_hier_barrier_knomial_start; */
    (*request) = (tccl_coll_req_h)req;
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
    tccl_hier_collreq_t *req = (tccl_hier_collreq_t *)request;
    return req->start(req);
}

static tccl_status_t tccl_hier_collective_wait(tccl_coll_req_h request)
{
    tccl_hier_collreq_t *req = (tccl_hier_collreq_t *)request;
    tccl_status_t status;
    while (TCCL_INPROGRESS == req->complete) {
        if (TCCL_OK != (status = req->progress(req))) {
            return status;
        };
    }
    assert(TCCL_OK == req->complete);
    return TCCL_OK;
}

static tccl_status_t tccl_hier_collective_test(tccl_coll_req_h request)
{
    tccl_hier_collreq_t *req = (tccl_hier_collreq_t *)request;
    tccl_status_t status;
    if (TCCL_INPROGRESS == req->complete) {
        if (TCCL_OK != (status = req->progress(req))) {
            return status;
        };
    }
    return req->complete;
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
    /* .super.team_create_post     = tccl_hier_team_create_post, */
    /* .super.team_destroy         = tccl_hier_team_destroy, */
    .super.progress             = NULL,
    .super.collective_init      = tccl_hier_collective_init,
    .super.collective_post      = tccl_hier_collective_post,
    .super.collective_wait      = tccl_hier_collective_wait,
    .super.collective_test      = tccl_hier_collective_test,
    .super.collective_finalize  = tccl_hier_collective_finalize
};
