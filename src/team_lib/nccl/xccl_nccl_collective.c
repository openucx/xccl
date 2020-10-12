/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <xccl_nccl_collective.h>
#include <cuda.h>

ncclDataType_t xccl_to_nccl_dtype[XCCL_DT_LAST_PREDEFINED];
ncclRedOp_t    xccl_to_nccl_reduce_op[XCCL_OP_LAST_PREDEFINED];

xccl_status_t
xccl_nccl_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_nccl_coll_req_t **request,
                               xccl_nccl_team_t *team)
{
    xccl_nccl_team_t *nccl_team       = ucs_derived_of(team, xccl_nccl_team_t);
    unsigned int     cuda_event_flags = cudaEventBlockingSync |
                                        cudaEventDisableTiming;

    *request = (xccl_nccl_coll_req_t*)malloc(sizeof(xccl_nccl_coll_req_t));
    if (*request == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    memcpy(&((*request)->args), coll_args, sizeof(xccl_coll_op_args_t));
    (*request)->team      = nccl_team;
    (*request)->super.lib = &xccl_team_lib_nccl.super;
    CUDACHECK(cudaEventCreateWithFlags(&((*request)->completed), cuda_event_flags));

    return XCCL_OK;
}

xccl_status_t
xccl_nccl_allreduce_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_coll_op_args_t  *args = &req->args;
    ncclResult_t nccl_st;

    nccl_st = ncclAllReduce(args->buffer_info.src_buffer,
                            args->buffer_info.dst_buffer,
                            args->reduce_info.count,
                            xccl_to_nccl_dtype[args->reduce_info.dt],
                            xccl_to_nccl_reduce_op[args->reduce_info.op],
                            req->team->nccl_comm,
                            req->team->stream);
    if (nccl_st != ncclSuccess) {
        xccl_nccl_error("ncclAllReduce failed (%d)", nccl_st);
        return XCCL_ERR_NO_MESSAGE;
    }
}

xccl_status_t
xccl_nccl_allreduce_init(xccl_coll_op_args_t *coll_args,
                         xccl_nccl_coll_req_t *request,
                         xccl_nccl_team_t *team)
{
    ncclRedOp_t          nccl_redop;
    ncclDataType_t       nccl_dt;

    nccl_redop = xccl_to_nccl_reduce_op[coll_args->reduce_info.op];
    nccl_dt    = xccl_to_nccl_dtype[coll_args->reduce_info.dt];
    if ((nccl_redop == ncclOpUnsupported) ||
        (nccl_dt    == ncclDataTypeUnsupported))
    {
        return XCCL_ERR_UNSUPPORTED;
    }

    request->coll_start = xccl_nccl_allreduce_start;
    return XCCL_OK;
}

xccl_status_t
xccl_nccl_alltoall_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_coll_op_args_t  *args = &req->args;
    ptrdiff_t sbuf, rbuf;
    size_t    data_size;
    int       group_size;
    int       peer;

    NCCLCHECK(ncclCommCount(req->team->nccl_comm, &group_size));
    sbuf      = (ptrdiff_t)args->buffer_info.src_buffer;
    rbuf      = (ptrdiff_t)args->buffer_info.dst_buffer;
    data_size = args->buffer_info.len;

    NCCLCHECK(ncclGroupStart());
    for (peer = 0; peer < group_size; peer++) {
      NCCLCHECK(ncclSend((void*)(sbuf + peer*data_size),
                          data_size, ncclChar, peer,
                          req->team->nccl_comm,
                          req->team->stream));
      NCCLCHECK(ncclRecv((void*)(rbuf + peer*data_size),
                          data_size, ncclChar, peer,
                          req->team->nccl_comm,
                          req->team->stream));

    }
    NCCLCHECK(ncclGroupEnd());

    return XCCL_OK;
}

xccl_status_t
xccl_nccl_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_nccl_coll_req_t *request,
                        xccl_nccl_team_t *team)
{
    request->coll_start = xccl_nccl_alltoall_start;
    return XCCL_OK;
}
