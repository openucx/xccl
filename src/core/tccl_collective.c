/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "tccl_team_lib.h"
tccl_status_t tccl_collective_init(tccl_coll_op_args_t *coll_args,
                                 tccl_coll_req_h *request, tccl_team_h team)
{
    return team->ctx->lib->collective_init(coll_args, request, team);
}

tccl_status_t tccl_collective_post(tccl_coll_req_h request)
{
    return request->lib->collective_post(request);
}

tccl_status_t tccl_collective_wait(tccl_coll_req_h request)
{
        return request->lib->collective_wait(request);
}

tccl_status_t tccl_collective_test(tccl_coll_req_h request)
{
        return request->lib->collective_test(request);
}

tccl_status_t tccl_collective_finalize(tccl_coll_req_h request)
{
    return request->lib->collective_finalize(request);
}

tccl_status_t tccl_context_progress(tccl_context_h team_ctx)
{
    if (team_ctx->lib->progress) {
        return team_ctx->lib->progress(team_ctx);
    }
    return TCCL_OK;
}
