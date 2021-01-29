/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <xccl_nccl_collective.h>
#include <cuda.h>

ncclDataType_t xccl_to_nccl_dtype[XCCL_DT_LAST_PREDEFINED];
ncclRedOp_t    xccl_to_nccl_reduce_op[XCCL_OP_LAST_PREDEFINED];

static inline xccl_status_t
xccl_nccl_get_free_status(xccl_nccl_team_t *team, xccl_cuda_status_t **status)
{
    int i;
    xccl_cuda_status_t *st;
    for (i = 0; i < STATUS_POOL_SIZE; i++) {
        st = team->status_pool + i;
        if (st->is_free) {
            st->is_free = 0;
            *status = st;
            return XCCL_OK;
        }
    }
    return XCCL_ERR_NO_RESOURCE;
}

xccl_status_t
xccl_nccl_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_nccl_coll_req_t **request,
                               xccl_nccl_team_t *team)
{
    xccl_nccl_team_t *nccl_team       = ucs_derived_of(team, xccl_nccl_team_t);
    unsigned int     cuda_event_flags = cudaEventDisableTiming;
    xccl_status_t st;

    *request = (xccl_nccl_coll_req_t*)malloc(sizeof(xccl_nccl_coll_req_t));
    if (*request == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    st = xccl_nccl_get_free_status(team, &((*request)->status));
    if (st != XCCL_OK) {
        free(*request);
        return st;
    }
    memcpy(&((*request)->args), coll_args, sizeof(xccl_coll_op_args_t));
    if (!(coll_args->field_mask & XCCL_COLL_OP_ARGS_FIELD_STREAM)) {
    /* use internal stream if stream is not provided via coll_args */
        xccl_nccl_trace("using internal stream");
        (*request)->args.stream.type   = XCCL_STREAM_TYPE_CUDA;
        (*request)->args.stream.stream = (void*)nccl_team->stream;
    }

    (*request)->team        = nccl_team;
    (*request)->super.lib   = &xccl_team_lib_nccl.super;
    (*request)->sync        = TEAM_NCCL_CTX_REQ(*request)->completion_sync;
    (*request)->barrier_buf = NULL;

    return XCCL_OK;
}

xccl_status_t
xccl_nccl_allreduce_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_coll_op_args_t  *args = &req->args;
    ncclResult_t nccl_st;
    cudaStream_t stream;

    stream = (cudaStream_t)args->stream.stream;
    nccl_st = ncclAllReduce(args->buffer_info.src_buffer,
                            args->buffer_info.dst_buffer,
                            args->reduce_info.count,
                            xccl_to_nccl_dtype[args->reduce_info.dt],
                            xccl_to_nccl_reduce_op[args->reduce_info.op],
                            req->team->nccl_comm,
                            stream);
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
    ptrdiff_t    sbuf, rbuf;
    size_t       data_size;
    int          group_size;
    int          peer;
    cudaStream_t stream;

    stream = (cudaStream_t)args->stream.stream;
    NCCLCHECK(ncclCommCount(req->team->nccl_comm, &group_size));
    sbuf      = (ptrdiff_t)args->buffer_info.src_buffer;
    rbuf      = (ptrdiff_t)args->buffer_info.dst_buffer;
    data_size = args->buffer_info.len;

    NCCLCHECK(ncclGroupStart());
    for (peer = 0; peer < group_size; peer++) {
        NCCLCHECK(ncclSend((void*)(sbuf + peer*data_size),
                           data_size, ncclChar, peer,
                           req->team->nccl_comm,
                           stream));
        NCCLCHECK(ncclRecv((void*)(rbuf + peer*data_size),
                           data_size, ncclChar, peer,
                           req->team->nccl_comm,
                           stream));

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

xccl_status_t
xccl_nccl_alltoallv_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_coll_op_args_t  *args = &req->args;
    ptrdiff_t    sbuf, rbuf;
    size_t       size, offset;
    int          send_dt_size, recv_dt_size;
    int          group_size, peer;
    cudaStream_t stream;

    stream = (cudaStream_t)args->stream.stream;
    NCCLCHECK(ncclCommCount(req->team->nccl_comm, &group_size));
    sbuf         = (ptrdiff_t)args->buffer_info.src_buffer;
    rbuf         = (ptrdiff_t)args->buffer_info.dst_buffer;
    send_dt_size = xccl_dt_size(args->buffer_info.src_datatype);
    recv_dt_size = xccl_dt_size(args->buffer_info.dst_datatype);

    NCCLCHECK(ncclGroupStart());
    for (peer = 0; peer < group_size; peer++) {
        offset = send_dt_size*args->buffer_info.src_displacements[peer];
        size   = send_dt_size*args->buffer_info.src_counts[peer];
        NCCLCHECK(ncclSend((void*)(sbuf + offset),
                           size, ncclChar, peer,
                           req->team->nccl_comm,
                           stream));

        offset = recv_dt_size*args->buffer_info.dst_displacements[peer];
        size   = recv_dt_size*args->buffer_info.dst_counts[peer];
        NCCLCHECK(ncclRecv((void*)(rbuf + offset),
                           size, ncclChar, peer,
                           req->team->nccl_comm,
                           stream));

    }
    NCCLCHECK(ncclGroupEnd());

    return XCCL_OK;
}


xccl_status_t
xccl_nccl_alltoallv_init(xccl_coll_op_args_t *coll_args,
                         xccl_nccl_coll_req_t *request,
                         xccl_nccl_team_t *team)
{
    request->coll_start = xccl_nccl_alltoallv_start;
    return XCCL_OK;
}

xccl_status_t
xccl_nccl_allgather_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_nccl_team_t     *team = ucs_derived_of(req->team, xccl_nccl_team_t);
    xccl_coll_op_args_t  *args = &req->args;
    ncclResult_t nccl_st;
    cudaStream_t stream;

    stream = (cudaStream_t)args->stream.stream;
    nccl_st = ncclAllGather(args->buffer_info.src_buffer,
                            args->buffer_info.dst_buffer,
                            args->buffer_info.len / team->team_size,
                            ncclChar,
                            req->team->nccl_comm,
                            stream);
    if (nccl_st != ncclSuccess) {
        xccl_nccl_error("ncclAllGather failed (%d)", nccl_st);
        return XCCL_ERR_NO_MESSAGE;
    }

    return XCCL_OK;
}

xccl_status_t
xccl_nccl_allgather_init(xccl_coll_op_args_t *coll_args,
                         xccl_nccl_coll_req_t *request,
                         xccl_nccl_team_t *team)
{
    request->coll_start = xccl_nccl_allgather_start;
    return XCCL_OK;
}

xccl_status_t
xccl_nccl_barrier_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_coll_op_args_t  *args = &req->args;
    ncclResult_t nccl_st;
    cudaStream_t stream;

    stream = (cudaStream_t)args->stream.stream;
    nccl_st = ncclAllReduce(req->barrier_buf, req->barrier_buf, 4, ncclFloat32,
                            ncclSum, req->team->nccl_comm, stream);
    if (nccl_st != ncclSuccess) {
        xccl_nccl_error("ncclBarrier failed (%d)", nccl_st);
        return XCCL_ERR_NO_MESSAGE;
    }
}

xccl_status_t
xccl_nccl_barrier_init(xccl_coll_op_args_t *coll_args,
                       xccl_nccl_coll_req_t *request,
                       xccl_nccl_team_t *team)
{
    CUDACHECK(cudaMalloc((void**)&request->barrier_buf, 4));
    request->coll_start = xccl_nccl_barrier_start;
    return XCCL_OK;
}

xccl_status_t
xccl_nccl_bcast_start(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_nccl_team_t     *team = ucs_derived_of(req->team, xccl_nccl_team_t);
    xccl_coll_op_args_t  *args = &req->args;
    ncclResult_t nccl_st;
    cudaStream_t stream;

    stream = (cudaStream_t)args->stream.stream;
    nccl_st = ncclBroadcast(args->buffer_info.src_buffer,
                            args->buffer_info.dst_buffer,
                            args->buffer_info.len,
                            ncclChar,
                            args->root,
                            req->team->nccl_comm,
                            stream);
    if (nccl_st != ncclSuccess) {
        xccl_nccl_error("ncclBroadcast failed (%d)", nccl_st);
        return XCCL_ERR_NO_MESSAGE;
    }

    return XCCL_OK;
}

xccl_status_t
xccl_nccl_bcast_init(xccl_coll_op_args_t *coll_args,
                     xccl_nccl_coll_req_t *request,
                     xccl_nccl_team_t *team)
{
    request->coll_start = xccl_nccl_bcast_start;
    return XCCL_OK;
}
