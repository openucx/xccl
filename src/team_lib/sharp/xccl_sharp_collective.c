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
    return UCS_OK;
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

static xccl_status_t
xccl_sharp_allreduce_init(xccl_coll_op_args_t* coll_args,
                          xccl_coll_req_h *request,
                          xccl_tl_team_t *team)
{
    xccl_sharp_coll_req_t *req            = malloc(sizeof(xccl_sharp_coll_req_t));
    xccl_sharp_team_t     *team_sharp     = xccl_derived_of(team, xccl_sharp_team_t);
    enum sharp_datatype   sharp_type      = xccl_to_sharp_dtype[coll_args->reduce_info.dt];
    enum sharp_reduce_op  sharp_op        = xccl_to_sharp_reduce_op[coll_args->reduce_info.op];
    xccl_sharp_context_t  *team_sharp_ctx = xccl_derived_of(team->ctx, xccl_sharp_context_t);
    struct sharp_coll_reduce_spec *reduce_spec;
    xccl_status_t                 status;
    void                          *src_mr, *dst_mr;
    int                           rc;

    if (sharp_type == SHARP_DTYPE_NULL || sharp_op == SHARP_OP_NULL) {
        return XCCL_ERR_NOT_IMPLEMENTED;
    }

    req->sharp_comm = team_sharp->sharp_comm;
    req->super.lib  = &xccl_team_lib_sharp.super;
    req->team       = team_sharp;
    
    if (team_sharp_ctx->rcache == NULL) {
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

    req->reduce_spec.sbuf_desc.buffer.ptr        = coll_args->buffer_info.src_buffer;
    req->reduce_spec.sbuf_desc.buffer.mem_handle = src_mr;
    req->reduce_spec.sbuf_desc.buffer.length     = coll_args->buffer_info.len;
    req->reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    req->reduce_spec.sbuf_desc.mem_type          = SHARP_MEM_TYPE_HOST;
 
    req->reduce_spec.rbuf_desc.buffer.ptr        = coll_args->buffer_info.dst_buffer;
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
    *request = &req->super;
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
    team_sharp      = xccl_derived_of(team, xccl_sharp_team_t);
    req->sharp_comm = team_sharp->sharp_comm;
    req->super.lib  = &xccl_team_lib_sharp.super;
    req->team       = team_sharp;
    req->start      = xccl_sharp_barrier_post;
    req->coll_type  = XCCL_BARRIER;

    *request = &req->super;
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
    xccl_sharp_coll_req_t *req = xccl_derived_of(request, xccl_sharp_coll_req_t);
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
    xccl_sharp_coll_req_t *req = xccl_derived_of(request, xccl_sharp_coll_req_t);
    int completed;

    completed = sharp_coll_req_test(req->handle);
    return (completed) ? XCCL_OK : XCCL_INPROGRESS;
}

xccl_status_t xccl_sharp_collective_finalize(xccl_coll_req_h request)
{
    xccl_sharp_coll_req_t *req  = xccl_derived_of(request, xccl_sharp_coll_req_t);
    xccl_sharp_team_t     *team = req->team;
    xccl_sharp_context_t  *ctx  = xccl_derived_of(team->super.ctx,
                                                  xccl_sharp_context_t);
    int           rc;

    sharp_coll_req_free(req->handle);
    if (req->coll_type != XCCL_BARRIER) {
        if (ctx->rcache == NULL) {
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
        } else {
            xccl_sharp_mem_rcache_dereg(ctx, req->src_rregion);
            xccl_sharp_mem_rcache_dereg(ctx, req->dst_rregion);
        }
    }
    free(request);
    return XCCL_OK;
}
