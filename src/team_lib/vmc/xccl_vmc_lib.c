/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "xccl_vmc_lib.h"
#include <stdio.h>
#include <vmc.h>

static int xccl_vmc_allgather(void *sbuf, void *rbuf, size_t len, void *comm)
{
    xccl_oob_collectives_t *oob =(xccl_oob_collectives_t*)comm;
    xccl_oob_allgather(sbuf, rbuf, len, oob);
    return 0;
}

void xccl_vmc_runtime_progress()
{
}

static xccl_status_t xccl_vmc_create_context(xccl_team_lib_t *lib,
                                             xccl_context_config_h config,
                                             xccl_tl_context_t **context)
{
    xccl_vmc_context_t *ctx = malloc(sizeof(*ctx));
    vmc_ctx_params_t    vmc_params = vmc_default_ctx_params;;
    vmc_status_t        st;
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);

    vmc_params.ib_dev_name           = NULL;
    vmc_params.mt_enabled            = 0;
    vmc_params.allgather             = xccl_vmc_allgather;
    vmc_params.timeout               = 10000;
    vmc_params.print_nack_stats      = 0;
    vmc_params.memtype_cache_enabled = 0;
    if (config->oob.size > 0) {
        vmc_params.world_size        = config->oob.size;
    }
    st = vmc_init(&vmc_params, &ctx->vmc_ctx);
    if (st != VMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize VMC collectives\n");
        free(ctx);
        return XCCL_ERR_NO_MESSAGE;
    }

    *context = &ctx->super;
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_destroy_context(xccl_tl_context_t *context)
{
    xccl_vmc_context_t *team_vmc_ctx = xccl_derived_of(context, xccl_vmc_context_t);
    if (team_vmc_ctx->vmc_ctx != NULL) {
        vmc_finalize(team_vmc_ctx->vmc_ctx);
    }
    free(team_vmc_ctx);
    return XCCL_OK;
}

static int vmc_comm_rank_to_world_mapper(int rank, void *mapper_ctx)
{
    xccl_team_config_h cfg = (xccl_team_config_h)mapper_ctx;
    return xccl_range_to_rank(cfg->range, rank);
}

static xccl_status_t xccl_vmc_team_create_post(xccl_tl_context_t *context,
                                               xccl_team_config_h config,
                                               xccl_oob_collectives_t oob,
                                               xccl_tl_team_t **team)
{
    xccl_vmc_context_t *team_vmc_ctx = xccl_derived_of(context, xccl_vmc_context_t);
    xccl_vmc_team_t    *team_vmc     = malloc(sizeof(*team_vmc));
    vmc_comm_params_t   vmc_params   = vmc_default_comm_params;
    vmc_status_t        st;
    XCCL_TEAM_SUPER_INIT(team_vmc->super, context, config, oob);
    vmc_params.comm_size            = oob.size;
    vmc_params.rank                 = oob.rank;
    vmc_params.comm_oob_context     = &team_vmc->super.oob;
    vmc_params.comm_rank_to_ctx     = vmc_comm_rank_to_world_mapper;
    vmc_params.rank_mapper_ctx      = (void*)&team_vmc->super.cfg;
    st = vmc_comm_create(team_vmc_ctx->vmc_ctx, &vmc_params, &team_vmc->vmc_comm);
    if (st != VMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize VMC collectives\n");
        free(team_vmc);
        return XCCL_ERR_NO_MESSAGE;
    }
    *team = &team_vmc->super;
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_team_create_test(xccl_tl_team_t *team)
{
    /*TODO implement true non-blocking */
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_team_destroy(xccl_tl_team_t *team)
{
    xccl_vmc_team_t *team_vmc = xccl_derived_of(team, xccl_vmc_team_t);
    if (team_vmc->vmc_comm != NULL) {
        vmc_comm_destroy(team_vmc->vmc_comm);
    }
    free(team);
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_collective_init(xccl_coll_op_args_t *coll_args,
                                              xccl_coll_req_h *request,
                                              xccl_tl_team_t *team)
{
    xccl_vmc_team_t     *team_vmc = xccl_derived_of(team, xccl_vmc_team_t);
    xccl_vmc_coll_req_t *req      = malloc(sizeof(*req));
    vmc_status_t              st;
    if (coll_args->coll_type != XCCL_BCAST) {
        return XCCL_ERR_UNSUPPORTED;
    }

    req->super.lib = &xccl_team_lib_vmc.super;
    req->buf       = coll_args->buffer_info.src_buffer;
    req->len       = coll_args->buffer_info.len;
    req->root      = coll_args->root;
    req->team      = team_vmc;

    *request = &req->super;
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_collective_post(xccl_coll_req_h request)
{
    xccl_vmc_coll_req_t *req = xccl_derived_of(request, xccl_vmc_coll_req_t);
    vmc_ibcast(req->buf, req->len, req->root, NULL, req->team->vmc_comm, &req->handle);
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_collective_test(xccl_coll_req_h request)
{
    xccl_vmc_coll_req_t *req = xccl_derived_of(request, xccl_vmc_coll_req_t);
    vmc_status_t st;
    st = vmc_req_test(req->handle);
    if (VMC_ERROR == st) {
        return XCCL_ERR_NO_MESSAGE;
    }
    return (st == VMC_SUCCESS)? XCCL_OK: XCCL_INPROGRESS;
}

static xccl_status_t xccl_vmc_collective_wait(xccl_coll_req_h request)
{
    xccl_status_t st = XCCL_INPROGRESS;
    while (st == XCCL_INPROGRESS) {
        st = xccl_vmc_collective_test(request);
    }
    return XCCL_OK;
}

static xccl_status_t xccl_vmc_collective_finalize(xccl_coll_req_h request)
{
    xccl_vmc_coll_req_t *req = xccl_derived_of(request, xccl_vmc_coll_req_t);
    vmc_req_free(req->handle);
    free(request);
    return XCCL_OK;
}

xccl_status_t xccl_vmc_context_progress(xccl_tl_context_t *context)
{
    xccl_vmc_context_t *team_vmc_ctx = xccl_derived_of(context, xccl_vmc_context_t);
    vmc_progress(team_vmc_ctx->vmc_ctx);
    return XCCL_OK;
}

xccl_team_lib_vmc_t xccl_team_lib_vmc = {
    .super.name                 = "vmc",
    .super.id                   = XCCL_TL_VMC,
    .super.priority             = 100,
    .super.params.reproducible  = XCCL_LIB_NON_REPRODUCIBLE,
    .super.params.thread_mode   = XCCL_LIB_THREAD_SINGLE | XCCL_LIB_THREAD_MULTIPLE,
    .super.params.team_usage    = XCCL_USAGE_HW_COLLECTIVES,
    .super.params.coll_types    = XCCL_COLL_CAP_BCAST,
    .super.ctx_create_mode      = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.create_team_context  = xccl_vmc_create_context,
    .super.destroy_team_context = xccl_vmc_destroy_context,
    .super.team_create_post     = xccl_vmc_team_create_post,
    .super.team_create_test     = xccl_vmc_team_create_test,
    .super.team_destroy         = xccl_vmc_team_destroy,
    .super.progress             = xccl_vmc_context_progress,
    .super.team_lib_open        = NULL,
    .super.collective_init      = xccl_vmc_collective_init,
    .super.collective_post      = xccl_vmc_collective_post,
    .super.collective_wait      = xccl_vmc_collective_wait,
    .super.collective_test      = xccl_vmc_collective_test,
    .super.collective_finalize  = xccl_vmc_collective_finalize,
    .super.global_mem_map_start = NULL,
    .super.global_mem_map_test  = NULL,
    .super.global_mem_unmap     = NULL,
};
