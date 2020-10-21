/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"
#include "xccl_mhba_collective.h"
#include "mem_component.h"
#include <ucs/memory/memory_type.h>
#include "core/xccl_team.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>

static ucs_config_field_t xccl_team_lib_mhba_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_team_lib_mhba_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t xccl_tl_mhba_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_mhba_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {NULL}
};

static xccl_status_t xccl_mhba_lib_open(xccl_team_lib_h self,
                                        xccl_team_lib_config_t *config)
{
    xccl_team_lib_mhba_t        *tl  = ucs_derived_of(self, xccl_team_lib_mhba_t);
    xccl_team_lib_mhba_config_t *cfg = ucs_derived_of(config, xccl_team_lib_mhba_config_t);

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_MHBA");
    xccl_mhba_debug("Team MHBA opened");
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_context_create(xccl_team_lib_h lib, xccl_context_params_t *params,
                         xccl_tl_context_config_t *config,
                         xccl_tl_context_t **context)
{
    xccl_mhba_context_t *ctx = malloc(sizeof(*ctx));

    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    *context = &ctx->super;

    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_context_destroy(xccl_tl_context_t *context)
{
    xccl_mhba_context_t *team_mhba_ctx =
        ucs_derived_of(context, xccl_mhba_context_t);

    free(team_mhba_ctx);

    return XCCL_OK;
}


static xccl_status_t
xccl_mhba_team_create_post(xccl_tl_context_t *context,
                           xccl_team_params_t *params,
                           xccl_team_t *base_team,
                           xccl_tl_team_t **team)
{
    xccl_mhba_team_t *mhba_team = malloc(sizeof(*mhba_team));
    xccl_sbgp_t *node;
    XCCL_TEAM_SUPER_INIT(mhba_team->super, context, params, base_team);
    node = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE);
    mhba_team->node.sbgp = node;
    size_t storage_size = (MHBA_CTRL_SIZE+MHBA_DATA_SIZE) * node->group_size;
    int shmid;
    if (0 == node->group_rank) {
        shmid = shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
    }
    xccl_sbgp_oob_bcast(&shmid, sizeof(int), 0, node, params->oob);
    if (shmid == -1) {
        xccl_mhba_error("failed to allocate sysv shm segment for %d bytes",
                        storage_size);
        return XCCL_ERR_NO_RESOURCE;
    }

    mhba_team->node.storage = shmat(shmid, NULL, 0);
    if (0 == node->group_rank) {
        if (shmctl(shmid, IPC_RMID, NULL) == -1) {
            xccl_mhba_error("failed to shmctl IPC_RMID seg %d",
                            shmid);
            return XCCL_ERR_NO_RESOURCE;
        }
    }
    if (mhba_team->node.storage == (void*)(-1)) {
        xccl_mhba_error("failed to shmat seg %d",
                        shmid);
        return XCCL_ERR_NO_RESOURCE;
    }
    mhba_team->node.ctrl = mhba_team->node.storage;
    mhba_team->node.umr_data = (void*)((ptrdiff_t)mhba_team->node.storage +
        node->group_size*MHBA_CTRL_SIZE);
    mhba_team->node.my_ctrl = (void*)((ptrdiff_t)mhba_team->node.ctrl +
        node->group_rank*MHBA_CTRL_SIZE);
    mhba_team->node.my_umr_data = (void*)((ptrdiff_t)mhba_team->node.umr_data +
        node->group_size*MHBA_DATA_SIZE);

    memset(mhba_team->node.my_ctrl, 0, MHBA_CTRL_SIZE);
    xccl_sbgp_oob_barrier(node, params->oob);
    *team = &mhba_team->super;
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_team_create_test(xccl_tl_team_t *team)
{
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_team_destroy(xccl_tl_team_t *team)
{
    xccl_mhba_team_t *mhba_team = ucs_derived_of(team, xccl_mhba_team_t);
    if (-1 == shmdt(mhba_team->node.storage)) {
        xccl_mhba_error("failed to shmdt %p, errno %d",
                        mhba_team->node.storage, errno);
    }
    free(team);
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_collective_init(xccl_coll_op_args_t *coll_args,
                          xccl_tl_coll_req_t **request,
                          xccl_tl_team_t *team)
{
    xccl_mhba_team_t *mhba_team  = ucs_derived_of(team, xccl_mhba_team_t);
    xccl_mhba_coll_req_t *req;
    xccl_status_t        status;
    ucs_memory_type_t    mem_type;

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer,
                                     &mem_type);
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

static xccl_status_t
xccl_mhba_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req  = ucs_derived_of(request, xccl_mhba_coll_req_t);
    xccl_status_t st;

    st = req->coll_start(request);
    if (st != XCCL_OK) {
        return st;
    }

    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req  = ucs_derived_of(request, xccl_mhba_coll_req_t);
    return XCCL_ERR_NOT_IMPLEMENTED;
}

static xccl_status_t
xccl_mhba_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req = ucs_derived_of(request, xccl_mhba_coll_req_t);
    free(req);
    return XCCL_OK;
}

xccl_team_lib_mhba_t xccl_team_lib_mhba = {
    .super.name                   = "mhba",
    .super.id                     = XCCL_TL_MHBA,
    .super.priority               = 90,
    .super.team_lib_config        =
    {
        .name                     = "MHBA team library",
        .prefix                   = "TEAM_MHBA_",
        .table                    = xccl_team_lib_mhba_config_table,
        .size                     = sizeof(xccl_team_lib_mhba_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "MHBA tl context",
        .prefix                  = "TEAM_MHBA_",
        .table                   = xccl_tl_mhba_context_config_table,
        .size                    = sizeof(xccl_tl_mhba_context_config_t),
    },
    .super.params.reproducible    = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode     = XCCL_THREAD_MODE_SINGLE | XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage      = XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    .super.params.coll_types      = XCCL_COLL_CAP_ALLTOALL,
    .super.mem_types              = UCS_BIT(UCS_MEMORY_TYPE_HOST),
    .super.ctx_create_mode        = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL,
    .super.team_context_create    = xccl_mhba_context_create,
    .super.team_context_destroy   = xccl_mhba_context_destroy,
    .super.team_context_progress  = NULL,
    .super.team_create_post       = xccl_mhba_team_create_post,
    .super.team_create_test       = xccl_mhba_team_create_test,
    .super.team_destroy           = xccl_mhba_team_destroy,
    .super.team_lib_open          = xccl_mhba_lib_open,
    .super.collective_init        = xccl_mhba_collective_init,
    .super.collective_post        = xccl_mhba_collective_post,
    .super.collective_wait        = NULL,
    .super.collective_test        = xccl_mhba_collective_test,
    .super.collective_finalize    = xccl_mhba_collective_finalize,
    .super.global_mem_map_start   = NULL,
    .super.global_mem_map_test    = NULL,
    .super.global_mem_unmap       = NULL,
};
