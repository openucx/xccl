/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include <xccl_collective.h>
#include <xccl_team.h>
#include <xccl_mm.h>
#include <utils/mem_component.h>
#include <ucs/memory/memory_type.h>
#include <stdlib.h>
#include <stdio.h>

xccl_status_t xccl_collective_init(xccl_coll_op_args_t *coll_args,
                                   xccl_coll_req_h *request, xccl_team_h team)
{
    XCCL_CHECK_TEAM(team);
    int               tl_team_id;
    xccl_tl_team_t    *tl_team;
    xccl_team_lib_t   *lib;
    xccl_coll_req_t   *xccl_req;
    xccl_status_t     status;
    ucs_memory_type_t mtype;

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer, &mtype);
    if (status != XCCL_OK) {
        xccl_error("Memtype detection error");
        return XCCL_ERR_INVALID_PARAM;
    }

    tl_team_id = team->coll_team_id[coll_args->coll_type][mtype];
    tl_team = team->tl_teams[tl_team_id];
    lib = tl_team->ctx->lib;
    xccl_trace("collective init: coll %d, team %s, memory type %s",
               coll_args->coll_type, lib->name, ucs_memory_type_names[mtype]);

    xccl_req = (xccl_coll_req_t*)malloc(sizeof(xccl_coll_req_t));
    if (xccl_req == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    status = lib->collective_init(coll_args, &xccl_req->req, tl_team);
    if (status != XCCL_OK) {
        free(xccl_req);
        return status;
    }

    *request = xccl_req;
    return status;
}

xccl_status_t xccl_collective_post(xccl_coll_req_h request)
{
    return request->req->lib->collective_post(request->req);
}

xccl_status_t xccl_collective_wait(xccl_coll_req_h request)
{
    return request->req->lib->collective_wait(request->req);
}

xccl_status_t xccl_collective_test(xccl_coll_req_h request)
{
    return request->req->lib->collective_test(request->req);
}

xccl_status_t xccl_collective_finalize(xccl_coll_req_h request)
{
    xccl_status_t status;

    status = request->req->lib->collective_finalize(request->req);
    free(request);
    return status;
}
