/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "tccl_vmc_lib.h"
#include <stdio.h>
#include <vmc.h>

static int tccl_vmc_allgather(void *sbuf, void *rbuf, size_t len, void *comm)
{
    tccl_oob_collectives_t *oob =(tccl_oob_collectives_t*)comm;
    return oob->allgather(sbuf, rbuf, len, oob->coll_context);
}

void tccl_vmc_runtime_progress()
{
}

static tccl_status_t tccl_vmc_create_context(tccl_team_lib_t *lib,
                                             tccl_context_config_h config,
                                             tccl_context_t **context)
{
    tccl_vmc_context_t *ctx = malloc(sizeof(*ctx));
    vmc_ctx_params_t    vmc_params;
    vmc_status_t        st;
    TCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);

    vmc_params.ib_dev_name           = NULL;
    vmc_params.mt_enabled            = 0;
    vmc_params.allgather             = tccl_vmc_allgather;
    vmc_params.timeout               = 10000;
    vmc_params.print_nack_stats      = 0;
    vmc_params.runtime_progress      = tccl_vmc_runtime_progress;
    vmc_params.memtype_cache_enabled = 0;
    if (config->oob.size > 0) {
        vmc_params.world_size        = config->oob.size;
    }
    st = vmc_init(&vmc_params, &ctx->vmc_ctx);
    if (st != VMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize VMC collectives\n");
        free(ctx);
        return TCCL_ERR_NO_MESSAGE;
    }

    *context = &ctx->super;
    return TCCL_OK;
}

static tccl_status_t tccl_vmc_destroy_context(tccl_context_h context)
{
    tccl_vmc_context_t *team_vmc_ctx = tccl_derived_of(context, tccl_vmc_context_t);
    if (team_vmc_ctx->vmc_ctx != NULL) {
        vmc_finalize(team_vmc_ctx->vmc_ctx);
    }
    free(team_vmc_ctx);
    return TCCL_OK;
}

static int vmc_comm_rank_to_world_mapper(int rank, void *mapper_ctx)
{
    tccl_team_config_h cfg = (tccl_team_config_h)mapper_ctx;
    return tccl_team_rank_to_world(cfg, rank);
}

static tccl_status_t tccl_vmc_team_create_post(tccl_context_h context,
                                               tccl_team_config_h config,
                                               tccl_oob_collectives_t oob,
                                               tccl_team_h *team)
{
    tccl_vmc_context_t *team_vmc_ctx = tccl_derived_of(context, tccl_vmc_context_t);
    tccl_vmc_team_t    *team_vmc     = malloc(sizeof(*team_vmc));
    vmc_comm_params_t   vmc_params;
    vmc_status_t        st;
    TCCL_TEAM_SUPER_INIT(team_vmc->super, context, config, oob);

    vmc_params.sx_depth             = 512;
    vmc_params.rx_depth             = 1024,
    vmc_params.sx_sge               = 1,
    vmc_params.rx_sge               = 2,
    vmc_params.post_recv_thresh     = 64,
    vmc_params.scq_moderation       = 64,
    vmc_params.wsize                = 64,
    vmc_params.cu_stage_thresh      = 4000,
    vmc_params.max_eager            = 65536,
    vmc_params.comm_size            = oob.size;
    vmc_params.rank                 = oob.rank;
    vmc_params.runtime_communicator = &team_vmc->super.oob;
    vmc_params.comm_rank_to_world   = vmc_comm_rank_to_world_mapper;
    vmc_params.rank_mapper_ctx      = (void*)&team_vmc->super.cfg;
    st = vmc_comm_create(team_vmc_ctx->vmc_ctx, &vmc_params, &team_vmc->vmc_comm);
    if (st != VMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize VMC collectives\n");
        free(team_vmc);
        return TCCL_ERR_NO_MESSAGE;
    }
    *team = &team_vmc->super;
    return TCCL_OK;
}

static tccl_status_t tccl_vmc_team_destroy(tccl_team_h team)
{
    tccl_vmc_team_t *team_vmc = tccl_derived_of(team, tccl_vmc_team_t);
    if (team_vmc->vmc_comm != NULL) {
        vmc_comm_destroy(team_vmc->vmc_comm);
    }
    free(team);
    return TCCL_OK;
}

static tccl_status_t tccl_vmc_collective_init(tccl_coll_op_args_t *coll_args,
                                              tccl_coll_req_h *request,
                                              tccl_team_h team)
{
    tccl_vmc_team_t     *team_vmc = tccl_derived_of(team, tccl_vmc_team_t);
    tccl_vmc_coll_req_t *req      = malloc(sizeof(*req));
    vmc_status_t              st;
    if (coll_args->coll_type != TCCL_BCAST) {
        return TCCL_ERR_UNSUPPORTED;
    }

    req->super.lib = team->ctx->lib;
    req->buf       = coll_args->buffer_info.src_buffer;
    req->len       = coll_args->buffer_info.len;
    req->root      = coll_args->root;
    req->team      = team_vmc;

    *request = &req->super;
    return TCCL_OK;
}

static tccl_status_t tccl_vmc_collective_post(tccl_coll_req_h request)
{
    tccl_vmc_coll_req_t *req = tccl_derived_of(request, tccl_vmc_coll_req_t);
    vmc_ibcast(req->buf, req->len, req->root, NULL, req->team->vmc_comm, &req->handle);
    return TCCL_OK;
}

static tccl_status_t tccl_vmc_collective_test(tccl_coll_req_h request)
{
    tccl_vmc_coll_req_t *req = tccl_derived_of(request, tccl_vmc_coll_req_t);
    vmc_status_t st;
    st = vmc_test(req->handle);
    if (VMC_ERROR == st) {
        return TCCL_ERR_NO_MESSAGE;
    }
    return (st == VMC_SUCCESS)? TCCL_OK: TCCL_INPROGRESS;
}

static tccl_status_t tccl_vmc_collective_wait(tccl_coll_req_h request)
{
    tccl_status_t st = TCCL_INPROGRESS;
    while (st == TCCL_INPROGRESS) {
        st = tccl_vmc_collective_test(request);
    }
    return TCCL_OK;
}

static tccl_status_t tccl_vmc_collective_finalize(tccl_coll_req_h request)
{
    tccl_vmc_coll_req_t *req = tccl_derived_of(request, tccl_vmc_coll_req_t);
    vmc_req_free(req->handle);
    free(request);
    return TCCL_OK;
}

tccl_status_t tccl_vmc_context_progress(tccl_context_h context)
{
    tccl_vmc_context_t *team_vmc_ctx = tccl_derived_of(context, tccl_vmc_context_t);
    vmc_progress(team_vmc_ctx->vmc_ctx);
    return TCCL_OK;
}

tccl_team_lib_vmc_t tccl_team_lib_vmc = {
    .super.name                 = "vmc",
    .super.priority             = 10,
    .super.config.reproducible  = TCCL_LIB_NON_REPRODUCIBLE,
    .super.config.thread_mode   = TCCL_LIB_THREAD_SINGLE | TCCL_LIB_THREAD_MULTIPLE,
    .super.config.team_usage    = TCCL_USAGE_HW_COLLECTIVES,
    .super.config.coll_types    = TCCL_BCAST,
    .super.ctx_create_mode      = TCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.create_team_context  = tccl_vmc_create_context,
    .super.destroy_team_context = tccl_vmc_destroy_context,
    .super.team_create_post     = tccl_vmc_team_create_post,
    .super.team_destroy         = tccl_vmc_team_destroy,
    .super.progress             = tccl_vmc_context_progress,
    .super.collective_init      = tccl_vmc_collective_init,
    .super.collective_post      = tccl_vmc_collective_post,
    .super.collective_wait      = tccl_vmc_collective_wait,
    .super.collective_test      = tccl_vmc_collective_test,
    .super.collective_finalize  = tccl_vmc_collective_finalize
};
