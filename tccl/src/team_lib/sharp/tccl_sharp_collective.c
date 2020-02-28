/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "tccl_sharp_collective.h"
#include "tccl_sharp_map.h"
#include <stdlib.h>
#include <string.h>

static inline tccl_status_t
tccl_sharp_get_free_buf(tccl_sharp_team_t *team_sharp,
                        tccl_sharp_buf_t **buffer)
{
    tccl_sharp_buf_t *buf;
    int i;

    for(i = 0; i < TCCL_SHARP_REG_BUF_NUM; i++) {
        buf = team_sharp->bufs + i;
        if (buf->used == 0) {
            buf->used = 1;
            *buffer = buf;
            return TCCL_OK;
        }
    }
    return TCCL_ERR_NO_RESOURCE;
}

static int
tccl_sharp_allreduce_post(tccl_sharp_coll_req_t *req)
{
    return sharp_coll_do_allreduce_nb(req->sharp_comm,
                                      &req->reduce_spec, &req->handle);
}

static tccl_status_t
tccl_sharp_allreduce_init(tccl_coll_op_args_t* coll_args,
                          tccl_coll_req_h *request,
                          tccl_team_h team)
{
    tccl_sharp_coll_req_t *req            = malloc(sizeof(tccl_sharp_coll_req_t));
    tccl_sharp_team_t     *team_sharp     = ucs_derived_of(team, tccl_sharp_team_t);
    enum sharp_datatype    sharp_type     = tccl_to_sharp_dtype[coll_args->reduce_info.dt];
    enum sharp_reduce_op   sharp_op       = tccl_to_sharp_reduce_op[coll_args->reduce_info.op];
    tccl_sharp_context_t  *team_sharp_ctx = ucs_derived_of(team->ctx, tccl_sharp_context_t);
    void *send_mr, *recv_mr;
    tccl_sharp_buf_t *sharp_buf;
    tccl_status_t use_free_buf;
    struct sharp_coll_reduce_spec *reduce_spec;
    int rc;

    if (sharp_type == SHARP_DTYPE_NULL || sharp_op == SHARP_OP_NULL) {
        return TCCL_ERR_NOT_IMPLEMENTED;
    }

    req->sharp_comm = team_sharp->sharp_comm;
    req->super.lib  = team->ctx->lib;
    req->team       = team_sharp;

    if (coll_args->buffer_info.len <= TCCL_SHARP_REG_BUF_SIZE) {
        use_free_buf = tccl_sharp_get_free_buf(team_sharp, &sharp_buf);
    } else {
        use_free_buf = TCCL_ERR_NO_RESOURCE;
    }

    if (use_free_buf == TCCL_OK) {
        memcpy(sharp_buf->buf, coll_args->buffer_info.src_buffer,
               coll_args->buffer_info.len);
        req->reduce_spec.sbuf_desc.buffer.ptr        = sharp_buf->buf;
        req->reduce_spec.sbuf_desc.buffer.mem_handle = sharp_buf->mr;
        req->reduce_spec.rbuf_desc.buffer.ptr        = sharp_buf->buf +
                                                       TCCL_SHARP_REG_BUF_SIZE;
        req->reduce_spec.rbuf_desc.buffer.mem_handle = sharp_buf->mr;
        req->sharp_buf                               = sharp_buf;
        req->sharp_buf->orig_src_buf                 = coll_args->buffer_info.src_buffer;
        req->sharp_buf->orig_dst_buf                 = coll_args->buffer_info.dst_buffer;
    } else {
        rc = sharp_coll_reg_mr(team_sharp_ctx->sharp_context,
                               coll_args->buffer_info.src_buffer,
                               coll_args->buffer_info.len, &send_mr);
        if (rc != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "SHARP regmr failed\n");
        }
        rc = sharp_coll_reg_mr(team_sharp_ctx->sharp_context,
                               coll_args->buffer_info.dst_buffer,
                               coll_args->buffer_info.len, &recv_mr);
        if (rc != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "SHARP regmr failed\n");
        }
        req->reduce_spec.sbuf_desc.buffer.ptr        = coll_args->buffer_info.src_buffer;
        req->reduce_spec.sbuf_desc.buffer.mem_handle = send_mr;
        req->reduce_spec.rbuf_desc.buffer.ptr        = coll_args->buffer_info.dst_buffer;
        req->reduce_spec.rbuf_desc.buffer.mem_handle = recv_mr;
        req->sharp_buf                               = NULL;
    }

    req->reduce_spec.sbuf_desc.buffer.length = coll_args->buffer_info.len;
    req->reduce_spec.sbuf_desc.type          = SHARP_DATA_BUFFER;
    req->reduce_spec.sbuf_desc.mem_type      = SHARP_MEM_TYPE_HOST;
    req->reduce_spec.rbuf_desc.buffer.length = coll_args->buffer_info.len;
    req->reduce_spec.rbuf_desc.type          = SHARP_DATA_BUFFER;
    req->reduce_spec.rbuf_desc.mem_type      = SHARP_MEM_TYPE_HOST;
    req->reduce_spec.length                  = coll_args->reduce_info.count;
    req->reduce_spec.dtype                   = sharp_type;
    req->reduce_spec.op                      = sharp_op;
    req->start                               = tccl_sharp_allreduce_post;
    req->coll_type                           = TCCL_ALLREDUCE;
#if HAVE_STRUCT_SHARP_COLL_REDUCE_SPEC_AGGR_MODE
    req->reduce_spec.aggr_mode               = SHARP_AGGREGATION_NONE;
#endif
    *request = &req->super;
    return TCCL_OK;
}

static int
tccl_sharp_barrier_post(tccl_sharp_coll_req_t *req)
{
    return sharp_coll_do_barrier_nb(req->sharp_comm, &req->handle);
}

static tccl_status_t
tccl_sharp_barrier_init(tccl_coll_op_args_t *coll_args,
                        tccl_coll_req_h *request,
                        tccl_team_h team)
{
    tccl_sharp_coll_req_t* req;
    tccl_sharp_team_t*     team_sharp;

    req             = malloc(sizeof(tccl_sharp_coll_req_t));
    team_sharp      = ucs_derived_of(team, tccl_sharp_team_t);
    req->sharp_comm = team_sharp->sharp_comm;
    req->super.lib  = team->ctx->lib;
    req->team       = team_sharp;
    req->start      = tccl_sharp_barrier_post;
    req->sharp_buf  = NULL;
    req->coll_type  = TCCL_BARRIER;

    *request = &req->super;
    return TCCL_OK;
}

tccl_status_t tccl_sharp_collective_init(tccl_coll_op_args_t *coll_args,
                                         tccl_coll_req_h *request,
                                         tccl_team_h team)
{
    switch(coll_args->coll_type) {
    case TCCL_ALLREDUCE:
        return tccl_sharp_allreduce_init(coll_args, request, team);
    case TCCL_BARRIER:
        return tccl_sharp_barrier_init(coll_args, request, team);
    default:
        break;
    }
    return TCCL_ERR_INVALID_PARAM;
}

tccl_status_t tccl_sharp_collective_post(tccl_coll_req_h request)
{
    tccl_sharp_coll_req_t *req = ucs_derived_of(request, tccl_sharp_coll_req_t);
    return (SHARP_COLL_SUCCESS == req->start(req)) ? TCCL_OK : TCCL_ERR_NO_MESSAGE;
}

tccl_status_t tccl_sharp_collective_wait(tccl_coll_req_h request)
{
    tccl_status_t              status = TCCL_INPROGRESS;
    while(status == TCCL_INPROGRESS) {
        status = tccl_sharp_collective_test(request);
    }
    return TCCL_OK;
}

tccl_status_t tccl_sharp_collective_test(tccl_coll_req_h request)
{
    tccl_sharp_coll_req_t *req = ucs_derived_of(request, tccl_sharp_coll_req_t);
    int completed;

    completed = sharp_coll_req_test(req->handle);
    if ((req->coll_type != TCCL_BARRIER) && (completed == 1) &&
        (req->sharp_buf != NULL)) {
        memcpy(req->sharp_buf->orig_dst_buf,
                req->reduce_spec.rbuf_desc.buffer.ptr,
                req->reduce_spec.rbuf_desc.buffer.length);
    }
    return (completed) ? TCCL_OK : TCCL_INPROGRESS;
}

tccl_status_t tccl_sharp_collective_finalize(tccl_coll_req_h request)
{
    tccl_sharp_coll_req_t *req  = ucs_derived_of(request, tccl_sharp_coll_req_t);
    tccl_sharp_team_t     *team = req->team;
    tccl_sharp_context_t  *ctx  = ucs_derived_of(team->super.ctx,
                                                  tccl_sharp_context_t);
    int rc;

    sharp_coll_req_free(req->handle);
    if (req->coll_type != TCCL_BARRIER) {
        if (req->sharp_buf == NULL) {
            rc = sharp_coll_dereg_mr(ctx->sharp_context,
                                    req->reduce_spec.sbuf_desc.buffer.mem_handle);
            if (rc != SHARP_COLL_SUCCESS) {
                fprintf(stderr, "SHARP deregmr failed\n");
            }

            rc = sharp_coll_dereg_mr(ctx->sharp_context,
                                    req->reduce_spec.rbuf_desc.buffer.mem_handle);
            if (rc != SHARP_COLL_SUCCESS) {
                fprintf(stderr, "SHARP deregmr failed\n");
            }

        } else {
            req->sharp_buf->used = 0;
        }
    }
    free(request);
    return TCCL_OK;
}
