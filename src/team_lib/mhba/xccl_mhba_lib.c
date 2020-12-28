/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"
#include "xccl_mhba_collective.h"
#include "xccl_mhba_ib.h"

#include <ucs/memory/memory_type.h>

static ucs_config_field_t xccl_team_lib_mhba_config_table[] = {
    {"", "", NULL, ucs_offsetof(xccl_team_lib_mhba_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)},

    {NULL}
};

static ucs_config_field_t xccl_tl_mhba_context_config_table[] = {
    {"", "", NULL, ucs_offsetof(xccl_tl_mhba_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)},

    {"NET_DEVICES", "", "Specifies which network device(s) to use",
     ucs_offsetof(xccl_tl_mhba_context_config_t, devices),
     UCS_CONFIG_TYPE_STRING_ARRAY},

    {"TRANSPOSE", "1", "Boolean - with transpose or not",
     ucs_offsetof(xccl_tl_mhba_context_config_t, transpose),
     UCS_CONFIG_TYPE_UINT},

    {"TRANSPOSE_HW_LIMITATIONS", "0",
     "Boolean - with transpose hw limitations or not",
     ucs_offsetof(xccl_tl_mhba_context_config_t, transpose_hw_limitations),
     UCS_CONFIG_TYPE_UINT}, //todo change to 1 in production

    {"IB_GLOBAL", "0", "Use global ib routing",
     ucs_offsetof(xccl_tl_mhba_context_config_t, ib_global),
     UCS_CONFIG_TYPE_UINT},

    {"TRANPOSE_BUF_SIZE", "128k", "Size of the pre-allocated transpose buffer",
     ucs_offsetof(xccl_tl_mhba_context_config_t, transpose_buf_size),
     UCS_CONFIG_TYPE_MEMUNITS},

    {"BLOCK_SIZE", "0", "Size of the blocks that are sent using blocked AlltoAll Algorithm",
    ucs_offsetof(xccl_tl_mhba_context_config_t, block_size),
    UCS_CONFIG_TYPE_UINT},

    {NULL}
};

static xccl_status_t xccl_mhba_lib_open(xccl_team_lib_h         self,
                                        xccl_team_lib_config_t *config)
{
    xccl_team_lib_mhba_t *tl = ucs_derived_of(self, xccl_team_lib_mhba_t);
    xccl_team_lib_mhba_config_t *cfg =
        ucs_derived_of(config, xccl_team_lib_mhba_config_t);

    tl->config.super.log_component.log_level =
        cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_MHBA");
    xccl_mhba_debug("Team MHBA opened");
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_context_create(xccl_team_lib_h           lib,
                                              xccl_context_params_t    *params,
                                              xccl_tl_context_config_t *config,
                                              xccl_tl_context_t       **context)
{
    xccl_tl_mhba_context_config_t *cfg =
        ucs_derived_of(config, xccl_tl_mhba_context_config_t);
    xccl_mhba_context_t *ctx        = malloc(sizeof(*ctx));
    char *               ib_devname = NULL;
    char                 tmp[128];
    int                  port = -1;

    if (!ctx) {
        xccl_mhba_error("context malloc faild");
        return XCCL_ERR_NO_MEMORY;
    }
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    if (cfg->devices.count > 0) {
        ib_devname        = cfg->devices.names[0];
        char *pos         = strstr(ib_devname, ":");
        int   devname_len = (int)(pos - ib_devname);
        strncpy(tmp, ib_devname, devname_len);
        tmp[devname_len] = '\0';
        ib_devname       = tmp;
        port             = atoi(pos + 1);
    }
    if (XCCL_OK != xccl_mhba_create_ibv_ctx(ib_devname, &ctx->ib_ctx)) {
        xccl_mhba_error("failed to allocate ibv_context");
        return XCCL_ERR_NO_MESSAGE;
    }
    if (port == -1) {
        port = xccl_mhba_get_active_port(ctx->ib_ctx);
    }
    ctx->ib_port = port;
    if (-1 == port || !xccl_mhba_check_port_active(ctx->ib_ctx, port)) {
        xccl_mhba_error("no active ports found on %s", ib_devname);
    }
    xccl_mhba_debug("using %s:%d", ib_devname, port);

    ctx->ib_pd = ibv_alloc_pd(ctx->ib_ctx);
    if (!ctx->ib_pd) {
        xccl_mhba_error("failed to allocate ib_pd");
        goto pd_alloc_failed;
    }
    memcpy(&ctx->cfg, cfg, sizeof(*cfg));
    *context = &ctx->super;

    return XCCL_OK;
pd_alloc_failed:
    ibv_close_device(ctx->ib_ctx);
    return XCCL_ERR_NO_MESSAGE;
}

static xccl_status_t xccl_mhba_context_destroy(xccl_tl_context_t *context)
{
    xccl_mhba_context_t *team_mhba_ctx =
        ucs_derived_of(context, xccl_mhba_context_t);
    if (ibv_dealloc_pd(team_mhba_ctx->ib_pd)) {
        xccl_mhba_error("Failed to dealloc PD errno %d", errno);
    }
    ibv_close_device(team_mhba_ctx->ib_ctx);
    free(team_mhba_ctx);
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_collective_init(xccl_coll_op_args_t *coll_args,
                                               xccl_tl_coll_req_t **request,
                                               xccl_tl_team_t      *team)
{
    xccl_mhba_team_t     *mhba_team = ucs_derived_of(team, xccl_mhba_team_t);
    xccl_mhba_coll_req_t *req;
    xccl_status_t         status;
    ucs_memory_type_t     mem_type;

    status =
        xccl_mem_component_type(coll_args->buffer_info.src_buffer, &mem_type);
    if (status != XCCL_OK) {
        xccl_mhba_error("Memtype detection error");
        return XCCL_ERR_INVALID_PARAM;
    }

    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
        return XCCL_ERR_UNSUPPORTED;
    }

    status = xccl_mhba_collective_init_base(coll_args, &req, mhba_team);
    if (status != XCCL_OK) {
        return status;
    }

    switch (coll_args->coll_type) {
    case XCCL_ALLTOALL:
        status = xccl_mhba_alltoall_init(coll_args, req, mhba_team);
        break;
    default:
        status = XCCL_ERR_INVALID_PARAM;
    }

    if (status != XCCL_OK) {
        free(req);
        return status;
    }

    (*request) = &req->super;
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req = ucs_derived_of(request, xccl_mhba_coll_req_t);
    xccl_schedule_start(&req->schedule);
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req = ucs_derived_of(request, xccl_mhba_coll_req_t);
    return req->schedule.super.state == XCCL_TASK_STATE_COMPLETED ?
        XCCL_OK : XCCL_INPROGRESS;
}

static xccl_status_t xccl_mhba_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_status_t         status = XCCL_OK;
    xccl_mhba_coll_req_t *req  = ucs_derived_of(request, xccl_mhba_coll_req_t);
    xccl_mhba_team_t     * team = req->team;
    if (ibv_dereg_mr(req->send_bf_mr)) {
        xccl_mhba_error("Failed to dereg_mr send buffer (errno=%d)", errno);
        status = XCCL_ERR_NO_MESSAGE;
    }
    if (ibv_dereg_mr(req->receive_bf_mr)) {
        xccl_mhba_error("Failed to dereg_mr send buffer (errno=%d)", errno);
        status = XCCL_ERR_NO_MESSAGE;
    }
    if (team->transpose) {
        free(req->tmp_transpose_buf);
        if (req->transpose_buf_mr != team->transpose_buf_mr) {
            ibv_dereg_mr(req->transpose_buf_mr);
            free(req->transpose_buf_mr->addr);
        }
    }
    free(req->tasks);
    free(req);
    return status;
}

xccl_team_lib_mhba_t xccl_team_lib_mhba = {
    .super.name     = "mhba",
    .super.id       = XCCL_TL_MHBA,
    .super.priority = 90,
    .super.team_lib_config =
        {
            .name   = "MHBA team library",
            .prefix = "TEAM_MHBA_",
            .table  = xccl_team_lib_mhba_config_table,
            .size   = sizeof(xccl_team_lib_mhba_config_t),
        },
    .super.tl_context_config =
        {
            .name   = "MHBA tl context",
            .prefix = "TEAM_MHBA_",
            .table  = xccl_tl_mhba_context_config_table,
            .size   = sizeof(xccl_tl_mhba_context_config_t),
        },
    .super.params.reproducible = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode =
        XCCL_THREAD_MODE_SINGLE | XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage     = XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    .super.params.coll_types     = XCCL_COLL_CAP_ALLTOALL,
    .super.mem_types             = UCS_BIT(UCS_MEMORY_TYPE_HOST),
    .super.ctx_create_mode       = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL,
    .super.team_context_create   = xccl_mhba_context_create,
    .super.team_context_destroy  = xccl_mhba_context_destroy,
    .super.team_context_progress = NULL,
    .super.team_create_post      = xccl_mhba_team_create_post,
    .super.team_create_test      = xccl_mhba_team_create_test,
    .super.team_destroy          = xccl_mhba_team_destroy,
    .super.team_lib_open         = xccl_mhba_lib_open,
    .super.collective_init       = xccl_mhba_collective_init,
    .super.collective_post       = xccl_mhba_collective_post,
    .super.collective_wait       = NULL,
    .super.collective_test       = xccl_mhba_collective_test,
    .super.collective_finalize   = xccl_mhba_collective_finalize,
    .super.global_mem_map_start  = NULL,
    .super.global_mem_map_test   = NULL,
    .super.global_mem_unmap      = NULL,
};
