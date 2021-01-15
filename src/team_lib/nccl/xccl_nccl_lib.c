/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_nccl_lib.h"
#include "xccl_nccl_collective.h"
#include "mem_component.h"
#include <ucs/memory/memory_type.h>

extern ncclDataType_t xccl_to_nccl_dtype[XCCL_DT_LAST_PREDEFINED];
extern ncclRedOp_t    xccl_to_nccl_reduce_op[XCCL_OP_LAST_PREDEFINED];

static void map_xccl_to_nccl_dtype()
{
    int dt;
    for (dt = 0; dt < XCCL_DT_LAST_PREDEFINED; dt++) {
        xccl_to_nccl_dtype[dt] = ncclDataTypeUnsupported;
    }
    xccl_to_nccl_dtype[XCCL_DT_INT8]    = ncclInt8;
    xccl_to_nccl_dtype[XCCL_DT_INT32]   = ncclInt32;
    xccl_to_nccl_dtype[XCCL_DT_INT64]   = ncclInt64;
    xccl_to_nccl_dtype[XCCL_DT_UINT8]   = ncclUint8;
    xccl_to_nccl_dtype[XCCL_DT_UINT32]  = ncclUint32;
    xccl_to_nccl_dtype[XCCL_DT_UINT64]  = ncclUint64;
    xccl_to_nccl_dtype[XCCL_DT_FLOAT16] = ncclFloat16;
    xccl_to_nccl_dtype[XCCL_DT_FLOAT32] = ncclFloat32;
    xccl_to_nccl_dtype[XCCL_DT_FLOAT64] = ncclFloat64;
}

static void map_xccl_to_nccl_reduce_op_type()
{
    int op;
    for (op = 0; op < XCCL_OP_LAST_PREDEFINED; op++) {
        xccl_to_nccl_reduce_op[op] = ncclOpUnsupported;
    }
    xccl_to_nccl_reduce_op[XCCL_OP_MAX]    = ncclMax;
    xccl_to_nccl_reduce_op[XCCL_OP_MIN]    = ncclMin;
    xccl_to_nccl_reduce_op[XCCL_OP_SUM]    = ncclSum;
    xccl_to_nccl_reduce_op[XCCL_OP_PROD]   = ncclProd;
}


static ucs_config_field_t xccl_team_lib_nccl_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_team_lib_nccl_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {"ALLREDUCE", "1",
     "Enable NCCL allreduce",
     ucs_offsetof(xccl_team_lib_nccl_config_t, enable_allreduce),
     UCS_CONFIG_TYPE_BOOL
    },

    {"ALLTOALL", "1",
     "Enable NCCL alltoall",
     ucs_offsetof(xccl_team_lib_nccl_config_t, enable_alltoall),
     UCS_CONFIG_TYPE_BOOL
    },

    {"ALLTOALLV", "1",
     "Enable NCCL alltoallv",
     ucs_offsetof(xccl_team_lib_nccl_config_t, enable_alltoallv),
     UCS_CONFIG_TYPE_BOOL
    },

    {"ALLGATHER", "1",
     "Enable NCCL allgather",
     ucs_offsetof(xccl_team_lib_nccl_config_t, enable_allgather),
     UCS_CONFIG_TYPE_BOOL
    },

    {"BCAST", "1",
     "Enable NCCL broadcast",
     ucs_offsetof(xccl_team_lib_nccl_config_t, enable_bcast),
     UCS_CONFIG_TYPE_BOOL
    },

    {NULL}
};

const char* xccl_nccl_sync_names[] = {
    [XCCL_NCCL_COMPLETION_SYNC_EVENT]    = "event",
    [XCCL_NCCL_COMPLETION_SYNC_CALLBACK] = "callback",
};


static ucs_config_field_t xccl_tl_nccl_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_nccl_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {"SYNC", "event",
     "Determines how XCCL tests completion of NCCL collective",
     ucs_offsetof(xccl_tl_nccl_context_config_t, completion_sync),
     UCS_CONFIG_TYPE_ENUM(xccl_nccl_sync_names)
    },

    {NULL}
};

static xccl_status_t xccl_nccl_lib_open(xccl_team_lib_h self,
                                        xccl_team_lib_config_t *config)
{
    xccl_team_lib_nccl_t        *tl  = ucs_derived_of(self, xccl_team_lib_nccl_t);
    xccl_team_lib_nccl_config_t *cfg = ucs_derived_of(config, xccl_team_lib_nccl_config_t);

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_NCCL");
    xccl_nccl_debug("Team NCCL opened");
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }
    if (cfg->enable_allreduce) {
        tl->super.params.coll_types |= XCCL_COLL_CAP_ALLREDUCE;
    }
    if (cfg->enable_alltoall) {
        tl->super.params.coll_types |= XCCL_COLL_CAP_ALLTOALL;
    }
    if (cfg->enable_alltoallv) {
        tl->super.params.coll_types |= XCCL_COLL_CAP_ALLTOALLV;
    }
    if (cfg->enable_allgather) {
        tl->super.params.coll_types |= XCCL_COLL_CAP_ALLGATHER;
    }
    if (cfg->enable_bcast) {
        tl->super.params.coll_types |= XCCL_COLL_CAP_BCAST;
    }

    map_xccl_to_nccl_dtype();
    map_xccl_to_nccl_reduce_op_type();

    return XCCL_OK;
}

static xccl_status_t
xccl_nccl_context_create(xccl_team_lib_h lib, xccl_context_params_t *params,
                         xccl_tl_context_config_t *config,
                         xccl_tl_context_t **context)
{
    xccl_nccl_context_t *ctx = malloc(sizeof(*ctx));
    xccl_tl_nccl_context_config_t *tl_config;

    if (ctx == NULL) {
        xccl_nccl_error("failed to allocate memory for nccl context");
        return XCCL_ERR_NO_MEMORY;
    }
    tl_config = ucs_derived_of(config, xccl_tl_nccl_context_config_t);
    ctx->completion_sync = tl_config->completion_sync;
    xccl_nccl_debug("sync type: %s", xccl_nccl_sync_names[ctx->completion_sync]);
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    *context = &ctx->super;

    return XCCL_OK;
}

static xccl_status_t
xccl_nccl_context_destroy(xccl_tl_context_t *context)
{
    xccl_nccl_context_t *team_nccl_ctx =
        ucs_derived_of(context, xccl_nccl_context_t);

    free(team_nccl_ctx);

    return XCCL_OK;
}

static xccl_status_t
xccl_nccl_team_create_post(xccl_tl_context_t *context,
                           xccl_team_params_t *params,
                           xccl_tl_team_t **team)
{
    xccl_nccl_team_t *nccl_team = malloc(sizeof(*nccl_team));
    ncclUniqueId unique_id, *gathered_ids;
    ncclResult_t nccl_st;
    int leastPriority=-1, greatestPriority=-1;

    XCCL_TEAM_SUPER_INIT(nccl_team->super, context, params);
    gathered_ids = (ncclUniqueId*)malloc(params->oob.size*sizeof(ncclUniqueId));

    if (params->oob.rank == 0) {
        nccl_st = ncclGetUniqueId(&unique_id);
        if (nccl_st != ncclSuccess) {
            xccl_nccl_error("ncclGetUniqueId failed (%d)", nccl_st);
            return XCCL_ERR_NO_MESSAGE;
        }
    }

    xccl_oob_allgather(&unique_id, gathered_ids, sizeof(ncclUniqueId), &params->oob);
    nccl_st = ncclCommInitRank(&nccl_team->nccl_comm,
                               params->oob.size,
                               gathered_ids[0],
                               params->oob.rank);
    free(gathered_ids);
    if (nccl_st != ncclSuccess) {
        /* Not a critical error in case we don't need GPU collectives */
        xccl_nccl_debug("ncclCommInitrank failed (%d)", nccl_st);
        return XCCL_ERR_NO_MESSAGE;
    }

    CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDACHECK(cudaStreamCreateWithPriority(&nccl_team->stream, cudaStreamNonBlocking, greatestPriority));
    nccl_team->team_size = params->oob.size;
    *team = &nccl_team->super;
    return XCCL_OK;
}

static xccl_status_t
xccl_nccl_team_create_test(xccl_tl_team_t *team)
{
    return XCCL_OK;
}

static xccl_status_t
xccl_nccl_team_destroy(xccl_tl_team_t *team)
{
    xccl_nccl_team_t *nccl_team = ucs_derived_of(team, xccl_nccl_team_t);
    if (nccl_team->nccl_comm != NULL) {
        ncclCommDestroy(nccl_team->nccl_comm);
    }
    if (nccl_team->stream != 0) {
        CUDACHECK(cudaStreamDestroy(nccl_team->stream));
    }
    free(team);
    return XCCL_OK;
}

static xccl_status_t
xccl_nccl_collective_init(xccl_coll_op_args_t *coll_args,
                          xccl_tl_coll_req_t **request,
                          xccl_tl_team_t *team)
{
    xccl_nccl_team_t *nccl_team  = ucs_derived_of(team, xccl_nccl_team_t);
    xccl_nccl_coll_req_t *req;
    xccl_status_t        status;
    ucs_memory_type_t    mem_type;
    ncclRedOp_t          nccl_redop;
    ncclDataType_t       nccl_dt;

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer,
                                     &mem_type);
    if (status != XCCL_OK) {
        xccl_nccl_error("Memtype detection error");
        return XCCL_ERR_INVALID_PARAM;
    }

    if (mem_type != UCS_MEMORY_TYPE_CUDA) {
        xccl_nccl_error("doesn't support memtype %d", mem_type);
        return XCCL_ERR_UNSUPPORTED;
    }

    if (!(UCS_BIT(coll_args->coll_type) & xccl_team_lib_nccl.super.params.coll_types)) {
        xccl_nccl_error("collective is not supported or disabled");
    }

    status = xccl_nccl_collective_init_base(coll_args, &req, nccl_team);
    if (status != XCCL_OK) {
        return status;
    }

    switch (coll_args->coll_type) {
    case XCCL_ALLREDUCE:
        status = xccl_nccl_allreduce_init(coll_args, req, nccl_team);
        break;
    case XCCL_ALLTOALL:
        status = xccl_nccl_alltoall_init(coll_args, req, nccl_team);
        break;
    case XCCL_ALLTOALLV:
        status = xccl_nccl_alltoallv_init(coll_args, req, nccl_team);
        break;
    case XCCL_ALLGATHER:
        status = xccl_nccl_allgather_init(coll_args, req, nccl_team);
        break;
    case XCCL_BCAST:
        status = xccl_nccl_bcast_init(coll_args, req, nccl_team);
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

static void nccl_completion_callback(void *request) {
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    req->status = XCCL_OK;
}

static xccl_status_t
xccl_nccl_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_status_t st;
    cudaStream_t *stream;

    st = req->coll_start(request);
    if (st != XCCL_OK) {
        xccl_nccl_error("collective start failed %d", st);
        return st;
    }

    if (req->completed != NULL) {
        st = xccl_mc_event_record(&req->args.stream, &req->completed);
    } else {
        stream = (cudaStream_t*)req->args.stream.stream;
        req->status = XCCL_INPROGRESS;
        CUDACHECK(cudaLaunchHostFunc(*stream, nccl_completion_callback, req));
        st = XCCL_OK;
    }

    return st;
}

static xccl_status_t
xccl_nccl_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_nccl_coll_req_t *req  = ucs_derived_of(request, xccl_nccl_coll_req_t);
    xccl_status_t st;

    if (req->completed != NULL) {
        /* use event to determine collective status */
        req->status = xccl_mc_event_query(req->completed);
        if (req->status != XCCL_INPROGRESS) {
            st = xccl_mc_event_free(req->completed);
            req->completed = NULL;
            if (st != XCCL_OK) {
                return st;
            }
        }
    }

    return req->status;
}

static xccl_status_t
xccl_nccl_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_status_t st;

    do {
        st = xccl_nccl_collective_test(request);
    } while (st == XCCL_INPROGRESS);

    return st;
}

static xccl_status_t
xccl_nccl_collective_finalize(xccl_tl_coll_req_t *request)
{
    free(request);

    return XCCL_OK;
}

xccl_team_lib_nccl_t xccl_team_lib_nccl = {
    .super.name                   = "nccl",
    .super.id                     = XCCL_TL_NCCL,
    .super.priority               = 90,
    .super.team_lib_config        =
    {
        .name                     = "NCCL team library",
        .prefix                   = "TEAM_NCCL_",
        .table                    = xccl_team_lib_nccl_config_table,
        .size                     = sizeof(xccl_team_lib_nccl_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "NCCL tl context",
        .prefix                  = "TEAM_NCCL_",
        .table                   = xccl_tl_nccl_context_config_table,
        .size                    = sizeof(xccl_tl_nccl_context_config_t),
    },
    .super.params.reproducible    = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode     = XCCL_THREAD_MODE_SINGLE | XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage      = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES,
/* supported collectives will be set in runtime */
    .super.params.coll_types      = 0,
    .super.mem_types              = UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode        = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create    = xccl_nccl_context_create,
    .super.team_context_destroy   = xccl_nccl_context_destroy,
    .super.team_context_progress  = NULL,
    .super.team_create_post       = xccl_nccl_team_create_post,
    .super.team_create_test       = xccl_nccl_team_create_test,
    .super.team_destroy           = xccl_nccl_team_destroy,
    .super.team_lib_open          = xccl_nccl_lib_open,
    .super.collective_init        = xccl_nccl_collective_init,
    .super.collective_post        = xccl_nccl_collective_post,
    .super.collective_wait        = xccl_nccl_collective_wait,
    .super.collective_test        = xccl_nccl_collective_test,
    .super.collective_finalize    = xccl_nccl_collective_finalize,
    .super.global_mem_map_start   = NULL,
    .super.global_mem_map_test    = NULL,
    .super.global_mem_unmap       = NULL,
};
