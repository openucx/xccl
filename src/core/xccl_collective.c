/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "xccl_team_lib.h"

xccl_status_t xccl_collective_init(xccl_coll_op_args_t *coll_args,
                                   xccl_coll_req_h *request, xccl_team_h team)
{
    int tl_team_id = team->coll_team_id[coll_args->coll_type];
    xccl_tl_team_t *tl_team = team->tl_teams[tl_team_id];
    xccl_team_lib_t *lib = tl_team->ctx->lib;
    return lib->collective_init(coll_args, request, tl_team);
}

xccl_status_t xccl_collective_post(xccl_coll_req_h request)
{
    return request->lib->collective_post(request);
}

xccl_status_t xccl_collective_wait(xccl_coll_req_h request)
{
    return request->lib->collective_wait(request);
}

xccl_status_t xccl_collective_test(xccl_coll_req_h request)
{
    return request->lib->collective_test(request);
}

xccl_status_t xccl_collective_finalize(xccl_coll_req_h request)
{
    return request->lib->collective_finalize(request);
}

xccl_status_t xccl_context_progress(xccl_context_h context)
{
    int i;
    xccl_tl_context_t *tl_ctx;
    for (i=0; i<context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        if (tl_ctx->lib->progress) {
            tl_ctx->lib->progress(tl_ctx);
        }
    }
    return XCCL_OK;
}
