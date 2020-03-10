/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "xccl_team_lib.h"
#include <stdlib.h>
#include <stdio.h>

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

xccl_status_t xccl_global_mem_map_start(xccl_team_h team, xccl_mem_map_params_t params,
                                        xccl_mem_h *memh_p)
{
    xccl_status_t status;
    int i;
    xccl_team_lib_t *tl;
    xccl_mem_handle_t *memh = calloc(1, sizeof(*memh) +
                                     sizeof(xccl_tl_mem_h)*(team->n_teams-1));
    for (i=0; i<team->n_teams; i++) {
        tl = team->tl_teams[i]->ctx->lib;
        if (tl->global_mem_map_start) {
            if (XCCL_OK != (status = tl->global_mem_map_start(
                                team->tl_teams[i], params, &memh->handles[i]))) {
                goto error;
            }
            memh->handles[i]->id = tl->id;
        }
    }
    memh->team = team;
    *memh_p = memh;
    return XCCL_OK;
error:
    *memh_p = NULL;
    free(memh);
    return status;
}

xccl_status_t xccl_global_mem_map_test(xccl_mem_h memh_p)
{
    xccl_status_t status;
    int all_done = 1;
    int i;
    xccl_team_lib_t *tl;
    xccl_mem_handle_t *memh = memh_p;
    for (i=0; i<memh->team->n_teams; i++) {
        tl = memh->team->tl_teams[i]->ctx->lib;
        if (memh->handles[i]) {
            assert(tl->global_mem_map_test);
            status = tl->global_mem_map_test(memh->handles[i]);
            if (XCCL_INPROGRESS == status) {
                all_done = 0;
            } else if (XCCL_OK != status) {
                return status;
            }
        }
    }
    return all_done == 1 ? XCCL_OK : XCCL_INPROGRESS;
}

xccl_status_t xccl_global_mem_unmap(xccl_mem_h memh_p)
{
    xccl_status_t status;
    int i;
    xccl_team_lib_t *tl;
    xccl_mem_handle_t *memh = memh_p;
    for (i=0; i<memh->team->n_teams; i++) {
        tl = memh->team->tl_teams[i]->ctx->lib;
        if (memh->handles[i]) {
            assert(tl->global_mem_unmap);
            if (XCCL_OK != (status = tl->global_mem_unmap(memh->handles[i]))) {
                return status;
            }
        }
    }
    return XCCL_OK;
}
