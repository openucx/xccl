/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "xccl_sharp_collective.h"
#include "xccl_sharp_map.h"
#include <stdlib.h>
#include <string.h>

static xccl_status_t
xccl_sharp_mem_rcache_reg(xccl_sharp_context_t *ctx, void *address,
                          size_t length, xccl_sharp_rcache_region_t **rregion)
{
    ucs_status_t        status;
    ucs_rcache_region_t *region;

    status = ucs_rcache_get(ctx->rcache, address, length, PROT_READ|PROT_WRITE,
                            NULL, &region);
    if (status != UCS_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }

    *rregion = (xccl_sharp_rcache_region_t*)region;
    return XCCL_OK;
}

static void xccl_sharp_mem_rcache_dereg(xccl_sharp_context_t *ctx,
                                        xccl_sharp_rcache_region_t *rregion)
{
    ucs_rcache_region_put(ctx->rcache, &rregion->super);
}

static int
xccl_sharp_allreduce_post(xccl_sharp_coll_req_t *req)
{
    return sharp_coll_do_allreduce_nb(req->sharp_comm,
                                      &req->reduce_spec, &req->handle);
}

static inline xccl_status_t
xccl_sharp_get_free_buf(xccl_sharp_team_t *team_sharp,
                        xccl_sharp_buf_t **buffer)
{
    unsigned bcopy_buf_num = xccl_team_lib_sharp.config.bcopy_buf_num;
    xccl_sharp_buf_t *buf;
    int              i;

    for(i = 0; i < bcopy_buf_num; i++) {
        buf = team_sharp->bufs + i;
        if (buf->used == 0) {
            buf->used = 1;
            *buffer = buf;
            return XCCL_OK;
        }
    }
    return XCCL_ERR_NO_RESOURCE;
}

static xccl_status_t
xccl_sharp_allreduce_init(xccl_coll_op_args_t* coll_args,
                          xccl_coll_req_h *request,
                          xccl_tl_team_t *team)
{
    xccl_sharp_coll_req_t *req            = malloc(sizeof(xccl_sharp_coll_req_t));
    xccl_sharp_team_t     *team_sharp     = ucs_derived_of(team, xccl_sharp_team_t);
    enum sharp_datatype   sharp_type      = xccl_to_sharp_dtype[coll_args->reduce_info.dt];
    enum sharp_reduce_op  sharp_op        = xccl_to_sharp_reduce_op[coll_args->reduce_info.op];
    xccl_sharp_context_t  *team_sharp_ctx = ucs_derived_of(team->ctx, xccl_sharp_context_t);
    unsigned              bcopy_buf_num   = xccl_team_lib_sharp.config.bcopy_buf_num;
    size_t                bcopy_buf_size  = xccl_team_lib_sharp.config.zcopy_thresh;

    struct sharp_coll_reduce_spec *reduce_spec;
    xccl_status_t                 status;
    void                          *src_mr, *dst_mr;
    void                          *src_buf, *dst_buf;
    int                           rc;
    xccl_status_t                 use_free_buf;
    xccl_sharp_buf_t              *sharp_buf;

    if (sharp_type == SHARP_DTYPE_NULL || sharp_op == SHARP_OP_NULL) {
        return XCCL_ERR_NOT_IMPLEMENTED;
    }

    req->sharp_comm = team_sharp->sharp_comm;
    req->super.lib  = &xccl_team_lib_sharp.super;
    req->team       = team_sharp;
    
    use_free_buf = XCCL_ERR_NO_RESOURCE;
    if (coll_args->buffer_info.len <= bcopy_buf_size) {
        use_free_buf =xccl_sharp_get_free_buf(team_sharp, &sharp_buf);
    }

    if (use_free_buf == XCCL_OK) {
        xccl_sharp_trace("sharp bcopy is used");
        memcpy(sharp_buf->buf, coll_args->buffer_info.src_buffer,
               coll_args->buffer_info.len);
        req->sharp_buf               = sharp_buf;
        req->sharp_buf->orig_src_buf = coll_args->buffer_info.src_buffer;
        req->sharp_buf->orig_dst_buf = coll_args->buffer_info.dst_buffer;
        src_buf = sharp_buf->buf;
        dst_buf = sharp_buf->buf + bcopy_buf_size;
        src_mr  = sharp_buf->mr;
        dst_mr  = sharp_buf->mr;
    }
    else {
        if (team_sharp_ctx->rcache == NULL) {
            xccl_sharp_trace("sharp direct registration is used");
            rc = sharp_coll_reg_mr(team_sharp_ctx->sharp_context,
                                coll_args->buffer_info.src_buffer,
                                coll_args->buffer_info.len, &src_mr);
            if (rc != SHARP_COLL_SUCCESS) {
                xccl_sharp_error("SHARP regmr failed for src buffer\n");
            }
            rc = sharp_coll_reg_mr(team_sharp_ctx->sharp_context,
                                coll_args->buffer_info.dst_buffer,
                                coll_args->buffer_info.len, &dst_mr);
            if (rc != SHARP_COLL_SUCCESS) {
                xccl_sharp_error("SHARP regmr failed for dst buffer\n");
            }
        } else {
            xccl_sharp_trace("sharp registration cache is used");
            status = xccl_sharp_mem_rcache_reg(team_sharp_ctx,
                                            coll_args->buffer_info.src_buffer,
                                            coll_args->buffer_info.len,
                                            &req->src_rregion);
            if (status != XCCL_OK) {
                xccl_sharp_error("SHARP regmr failed for src buffer\n");
            }
            src_mr = req->src_rregion->memh;

            status = xccl_sharp_mem_rcache_reg(team_sharp_ctx,
                                            coll_args->buffer_info.dst_buffer,
                                            coll_args->buffer_info.len,
                                            &req->dst_rregion);
            if (status != XCCL_OK) {
                xccl_sharp_error("SHARP regmr failed for dst buffer\n");
            }
            dst_mr = req->dst_rregion->memh;
        }
        src_buf = coll_args->buffer_info.src_buffer;
        dst_buf = coll_args->buffer_info.dst_buffer;
        req->sharp_buf = NULL;
    }

    req->reduce_spec.sbuf_desc.buffer.ptr        = src_buf;
    req->reduce_spec.sbuf_desc.buffer.mem_handle = src_mr;
    req->reduce_spec.sbuf_desc.buffer.length     = coll_args->buffer_info.len;
    req->reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    req->reduce_spec.sbuf_desc.mem_type          = SHARP_MEM_TYPE_HOST;
 
    req->reduce_spec.rbuf_desc.buffer.ptr        = dst_buf;
    req->reduce_spec.rbuf_desc.buffer.mem_handle = dst_mr;
    req->reduce_spec.rbuf_desc.buffer.length     = coll_args->buffer_info.len;
    req->reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    req->reduce_spec.rbuf_desc.mem_type          = SHARP_MEM_TYPE_HOST;
 
    req->reduce_spec.length                      = coll_args->reduce_info.count;
    req->reduce_spec.dtype                       = sharp_type;
    req->reduce_spec.op                          = sharp_op;
 
    req->start                                   = xccl_sharp_allreduce_post;
    req->coll_type                               = XCCL_ALLREDUCE;
#if HAVE_STRUCT_SHARP_COLL_REDUCE_SPEC_AGGR_MODE
    req->reduce_spec.aggr_mode                   = SHARP_AGGREGATION_NONE;
#endif
    *request = (xccl_coll_req_h)&req->super;
    return XCCL_OK;
}

static int
xccl_sharp_barrier_post(xccl_sharp_coll_req_t *req)
{
    return sharp_coll_do_barrier_nb(req->sharp_comm, &req->handle);
}

static xccl_status_t
xccl_sharp_barrier_init(xccl_coll_op_args_t *coll_args,
                        xccl_coll_req_h *request,
                        xccl_tl_team_t *team)
{
    xccl_sharp_coll_req_t* req;
    xccl_sharp_team_t*     team_sharp;

    req             = malloc(sizeof(xccl_sharp_coll_req_t));
    team_sharp      = ucs_derived_of(team, xccl_sharp_team_t);
    req->sharp_comm = team_sharp->sharp_comm;
    req->super.lib  = &xccl_team_lib_sharp.super;
    req->team       = team_sharp;
    req->start      = xccl_sharp_barrier_post;
    req->coll_type  = XCCL_BARRIER;
    req->sharp_buf  = NULL;
    
    *request = (xccl_coll_req_h)&req->super;
    return XCCL_OK;
}

xccl_status_t xccl_sharp_collective_init(xccl_coll_op_args_t *coll_args,
                                         xccl_coll_req_h *request,
                                         xccl_tl_team_t *team)
{
    switch(coll_args->coll_type) {
    case XCCL_ALLREDUCE:
        return xccl_sharp_allreduce_init(coll_args, request, team);
    case XCCL_BARRIER:
        return xccl_sharp_barrier_init(coll_args, request, team);
    default:
        break;
    }
    return XCCL_ERR_INVALID_PARAM;
}

xccl_status_t xccl_sharp_collective_post(xccl_coll_req_h request)
{
    xccl_sharp_coll_req_t *req = ucs_derived_of(request, xccl_sharp_coll_req_t);
    return (SHARP_COLL_SUCCESS == req->start(req)) ? XCCL_OK : XCCL_ERR_NO_MESSAGE;
}

xccl_status_t xccl_sharp_collective_wait(xccl_coll_req_h request)
{
    xccl_status_t              status = XCCL_INPROGRESS;
    while(status == XCCL_INPROGRESS) {
        status = xccl_sharp_collective_test(request);
    }
    return XCCL_OK;
}

xccl_status_t xccl_sharp_collective_test(xccl_coll_req_h request)
{
    xccl_sharp_coll_req_t *req = ucs_derived_of(request, xccl_sharp_coll_req_t);
    int completed;

    completed = sharp_coll_req_test(req->handle);
    if ((req->coll_type != XCCL_BARRIER) && (completed == 1) &&
        (req->sharp_buf != NULL)) {
        memcpy(req->sharp_buf->orig_dst_buf,
               req->reduce_spec.rbuf_desc.buffer.ptr,
               req->reduce_spec.rbuf_desc.buffer.length);
    }
    return (completed) ? XCCL_OK : XCCL_INPROGRESS;
}

xccl_status_t xccl_sharp_collective_finalize(xccl_coll_req_h request)
{
    xccl_sharp_coll_req_t *req  = ucs_derived_of(request, xccl_sharp_coll_req_t);
    xccl_sharp_team_t     *team = req->team;
    xccl_sharp_context_t  *ctx  = ucs_derived_of(team->super.ctx,
                                                  xccl_sharp_context_t);
    int rc;

    sharp_coll_req_free(req->handle);
    if (req->coll_type != XCCL_BARRIER) {
        if (req->sharp_buf != NULL){
            req->sharp_buf->used = 0;
        }
        else if (ctx->rcache != NULL) {
            xccl_sharp_mem_rcache_dereg(ctx, req->src_rregion);
            xccl_sharp_mem_rcache_dereg(ctx, req->dst_rregion);

        } else {
            rc = sharp_coll_dereg_mr(ctx->sharp_context,
                                     req->reduce_spec.sbuf_desc.buffer.mem_handle);
            if (rc != SHARP_COLL_SUCCESS) {
                xccl_sharp_error("SHARP deregmr failed\n");
            }

            rc = sharp_coll_dereg_mr(ctx->sharp_context,
                                     req->reduce_spec.rbuf_desc.buffer.mem_handle);
            if (rc != SHARP_COLL_SUCCESS) {
                xccl_sharp_error("SHARP deregmr failed\n");
            }
        }
    }
    free(request);
    return XCCL_OK;
}
