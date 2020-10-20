/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <xccl_mhba_collective.h>

xccl_status_t
xccl_mhba_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_mhba_coll_req_t **request,
                               xccl_mhba_team_t *team)
{
    xccl_mhba_team_t *mhba_team       = ucs_derived_of(team, xccl_mhba_team_t);
    *request = (xccl_mhba_coll_req_t*)malloc(sizeof(xccl_mhba_coll_req_t));
    if (*request == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    memcpy(&((*request)->args), coll_args, sizeof(xccl_coll_op_args_t));
    (*request)->team      = mhba_team;
    (*request)->super.lib = &xccl_team_lib_mhba.super;
    return XCCL_OK;
}

xccl_status_t
xccl_mhba_alltoall_start(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req  = ucs_derived_of(request, xccl_mhba_coll_req_t);
    xccl_coll_op_args_t  *args = &req->args;
    ptrdiff_t sbuf, rbuf;
    size_t    data_size;
    int       group_size;
    int       peer;

    return XCCL_ERR_NOT_IMPLEMENTED;
}

xccl_status_t
xccl_mhba_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_mhba_coll_req_t *request,
                        xccl_mhba_team_t *team)
{
    request->coll_start = xccl_mhba_alltoall_start;
    return XCCL_OK;
}
