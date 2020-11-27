/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "xccl_hmc_lib.h"
#include <stdio.h>
#include <hmc.h>
#include <ucs/memory/memory_type.h>

static ucs_config_field_t xccl_tl_hmc_context_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(xccl_tl_hmc_context_config_t, super),
        UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {NULL}
};

static int xccl_hmc_allgather(void *sbuf, void *rbuf, size_t len, void *comm)
{
    xccl_oob_collectives_t *oob =(xccl_oob_collectives_t*)comm;
    xccl_oob_allgather(sbuf, rbuf, len, oob);
    return 0;
}

void xccl_hmc_runtime_progress()
{
}

static xccl_status_t xccl_hmc_create_context(xccl_team_lib_t *lib,
                                             xccl_context_params_t *params,
                                             xccl_tl_context_config_t *config,
                                             xccl_tl_context_t **context)
{
    xccl_hmc_context_t *ctx = malloc(sizeof(*ctx));
    hmc_ctx_params_t    ctx_params;
    hmc_ctx_config_h    ctx_config;
    hmc_status_t        st;
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);

    ctx_params.field_mask = HMC_CTX_PARAMS_FIELD_WORLD_SIZE |
                            HMC_CTX_PARAMS_FIELD_ALLGATHER  |
                            HMC_CTX_PARAMS_FIELD_OOB_CONTEXT;

    ctx_params.allgather   = xccl_hmc_allgather;
    ctx_params.oob_context = (void*)(&ctx->super.params.oob);
    ctx_params.world_size  = params->oob.size;

    hmc_context_config_read(&ctx_config);
    st = hmc_init(&ctx_params, ctx_config, &ctx->hmc_ctx);
    hmc_context_config_release(ctx_config);

    if (st != HMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize HMC collectives\n");
        free(ctx);
        return XCCL_ERR_NO_MESSAGE;
    }

    *context = &ctx->super;
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_destroy_context(xccl_tl_context_t *context)
{
    xccl_hmc_context_t *team_hmc_ctx = ucs_derived_of(context, xccl_hmc_context_t);
    if (team_hmc_ctx->hmc_ctx != NULL) {
        hmc_finalize(team_hmc_ctx->hmc_ctx);
    }
    free(team_hmc_ctx);
    return XCCL_OK;
}

static int hmc_comm_rank_to_world_mapper(int rank, void *mapper_ctx)
{
    xccl_team_params_t *params = (xccl_team_params_t*)mapper_ctx;
    return xccl_range_to_rank(params->range, rank);
}

static xccl_status_t xccl_hmc_team_create_post(xccl_tl_context_t *context,
                                               xccl_team_params_t *params,
                                               xccl_tl_team_t **team)
{
    xccl_hmc_context_t *team_hmc_ctx = ucs_derived_of(context, xccl_hmc_context_t);
    xccl_hmc_team_t    *team_hmc     = malloc(sizeof(*team_hmc));
    hmc_comm_params_t   comm_params;
    hmc_status_t        st;
    XCCL_TEAM_SUPER_INIT(team_hmc->super, context, params);
    comm_params.field_mask = HMC_COMM_PARAMS_FIELD_COMM_SIZE        |
                             HMC_COMM_PARAMS_FIELD_COMM_RANK        |
                             HMC_COMM_PARAMS_FIELD_COMM_RANK_TO_CTX |
                             HMC_COMM_PARAMS_FIELD_RANK_MAPPER_CTX  |
                             HMC_COMM_PARAMS_FIELD_COMM_OOB_CONTEXT;

    comm_params.comm_size        = params->oob.size;
    comm_params.comm_rank        = params->oob.rank;
    comm_params.comm_oob_context = &team_hmc->super.params.oob;
    comm_params.comm_rank_to_ctx = hmc_comm_rank_to_world_mapper;
    comm_params.rank_mapper_ctx  = (void*)&team_hmc->super.params;
    st = hmc_comm_create(team_hmc_ctx->hmc_ctx, &comm_params, &team_hmc->hmc_comm);
    if (st != HMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize HMC collectives\n");
        free(team_hmc);
        return XCCL_ERR_NO_MESSAGE;
    }
    *team = &team_hmc->super;
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_team_create_test(xccl_tl_team_t *team)
{
    /*TODO implement true non-blocking */
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_team_destroy(xccl_tl_team_t *team)
{
    xccl_hmc_team_t *team_hmc = ucs_derived_of(team, xccl_hmc_team_t);
    if (team_hmc->hmc_comm != NULL) {
        hmc_comm_destroy(team_hmc->hmc_comm);
    }
    free(team);
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_collective_init(xccl_coll_op_args_t *coll_args,
                                              xccl_tl_coll_req_t **request,
                                              xccl_tl_team_t *team)
{
    xccl_hmc_team_t     *team_hmc = ucs_derived_of(team, xccl_hmc_team_t);
    xccl_hmc_coll_req_t *req      = malloc(sizeof(*req));
    hmc_status_t              st;
    if (coll_args->coll_type != XCCL_BCAST) {
        return XCCL_ERR_UNSUPPORTED;
    }

    req->super.lib = &xccl_team_lib_hmc.super;
    req->buf       = coll_args->buffer_info.src_buffer;
    req->len       = coll_args->buffer_info.len;
    req->root      = coll_args->root;
    req->team      = team_hmc;

    *request = (xccl_tl_coll_req_t*)&req->super;
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_hmc_coll_req_t *req = ucs_derived_of(request, xccl_hmc_coll_req_t);
    hmc_bcast_args_t args;
    args.field_mask = HMC_BCAST_ARGS_FIELD_ADDRESS |
                      HMC_BCAST_ARGS_FIELD_SIZE |
                      HMC_BCAST_ARGS_FIELD_ROOT |
                      HMC_BCAST_ARGS_FIELD_COMM;

    args.address = req->buf;
    args.size    = req->len;
    args.root    = req->root;
    args.comm    = req->team->hmc_comm;

    hmc_ibcast(&args, &req->handle);
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_hmc_coll_req_t *req = ucs_derived_of(request, xccl_hmc_coll_req_t);
    hmc_status_t st;
    st = hmc_req_test(req->handle);
    if (HMC_ERROR == st) {
        return XCCL_ERR_NO_MESSAGE;
    }
    return (st == HMC_SUCCESS)? XCCL_OK: XCCL_INPROGRESS;
}

static xccl_status_t xccl_hmc_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_status_t st = XCCL_INPROGRESS;
    while (st == XCCL_INPROGRESS) {
        st = xccl_hmc_collective_test(request);
    }
    return XCCL_OK;
}

static xccl_status_t xccl_hmc_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_hmc_coll_req_t *req = ucs_derived_of(request, xccl_hmc_coll_req_t);
    hmc_req_free(req->handle);
    free(request);
    return XCCL_OK;
}

xccl_status_t xccl_hmc_context_progress(xccl_tl_context_t *context)
{
    xccl_hmc_context_t *team_hmc_ctx = ucs_derived_of(context, xccl_hmc_context_t);
    hmc_progress(team_hmc_ctx->hmc_ctx);
    return XCCL_OK;
}

xccl_team_lib_hmc_t xccl_team_lib_hmc = {
    .super.name                  = "hmc",
    .super.id                    = XCCL_TL_HMC,
    .super.priority              = 100,
    .super.tl_context_config     = {
        .name                    = "HMC tl context",
        .prefix                  = "TEAM_HMC_",
        .table                   = xccl_tl_hmc_context_config_table,
        .size                    = sizeof(xccl_tl_hmc_context_config_t),
    },
    .super.params.reproducible   = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode    = XCCL_THREAD_MODE_SINGLE | XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage     = XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    .super.params.coll_types     = XCCL_COLL_CAP_BCAST,
    .super.mem_types             = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                   UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode       = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create   = xccl_hmc_create_context,
    .super.team_context_destroy  = xccl_hmc_destroy_context,
    .super.team_context_progress = xccl_hmc_context_progress,
    .super.team_create_post      = xccl_hmc_team_create_post,
    .super.team_create_test      = xccl_hmc_team_create_test,
    .super.team_destroy          = xccl_hmc_team_destroy,
    .super.team_lib_open         = NULL,
    .super.collective_init       = xccl_hmc_collective_init,
    .super.collective_post       = xccl_hmc_collective_post,
    .super.collective_wait       = xccl_hmc_collective_wait,
    .super.collective_test       = xccl_hmc_collective_test,
    .super.collective_finalize   = xccl_hmc_collective_finalize,
    .super.global_mem_map_start  = NULL,
    .super.global_mem_map_test   = NULL,
    .super.global_mem_unmap      = NULL,
};
