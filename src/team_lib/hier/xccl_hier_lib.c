/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_hier_lib.h"
#include "xccl_hier_team.h"
#include "xccl_hier_context.h"
#include "xccl_hier_schedule.h"
#include "xccl_hier_task_schedule.h"
#include "utils/mem_component.h"
#include <ucs/memory/memory_type.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t xccl_team_lib_hier_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(xccl_team_lib_hier_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};


static ucs_config_field_t xccl_tl_hier_context_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(xccl_tl_hier_context_config_t, super),
        UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {"NET_DEVICES", "all",
     "Specifies which network device(s) to use",
     ucs_offsetof(xccl_tl_hier_context_config_t, devices),
     UCS_CONFIG_TYPE_STRING_ARRAY
    },

    {"ENABLE_SHARP", "no",
     "Enables sharp team in hier team",
     ucs_offsetof(xccl_tl_hier_context_config_t, enable_sharp),
     UCS_CONFIG_TYPE_BOOL
    },

    {"ENABLE_SHMSEG", "no",
     "Enables shmseg team in hier team",
     ucs_offsetof(xccl_tl_hier_context_config_t, enable_shmseg),
     UCS_CONFIG_TYPE_BOOL
     },

    {"ENABLE_HMC", "no",
     "Enables hmc team in hier team",
     ucs_offsetof(xccl_tl_hier_context_config_t, enable_hmc),
     UCS_CONFIG_TYPE_BOOL
     },

    {"ENABLE_NCCL", "no",
     "Enables nccl team in hier team",
     ucs_offsetof(xccl_tl_hier_context_config_t, enable_nccl),
     UCS_CONFIG_TYPE_BOOL
     },

    {"BCAST_PIPELINE_THRESH", "inf",
     "",
     ucs_offsetof(xccl_tl_hier_context_config_t, bcast_pipeline_thresh),
     UCS_CONFIG_TYPE_MEMUNITS
     },

    {"BCAST_PIPELINE_DEPTH", "1",
     "",
     ucs_offsetof(xccl_tl_hier_context_config_t, bcast_pipeline_depth),
     UCS_CONFIG_TYPE_UINT
     },

    {"BCAST_SM_GET", "0",
     "",
     ucs_offsetof(xccl_tl_hier_context_config_t, bcast_sm_get),
     UCS_CONFIG_TYPE_BOOL
     },

    {"BCAST_SM_GET_THRESH", "inf",
     "",
     ucs_offsetof(xccl_tl_hier_context_config_t, bcast_sm_get_thresh),
     UCS_CONFIG_TYPE_MEMUNITS
     },

    {"NODE_LEADER_RANK_ID", "0",
     "",
     ucs_offsetof(xccl_tl_hier_context_config_t, node_leader_rank_id),
     UCS_CONFIG_TYPE_UINT
     },

    {NULL}
};

static xccl_status_t xccl_hier_open(xccl_team_lib_h self,
                                    xccl_team_lib_config_t *config)
{
    xccl_team_lib_hier_t *tl  = ucs_derived_of(self, xccl_team_lib_hier_t);

    tl->config.super.log_component.log_level = config->log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", tl->super.name);

    return XCCL_OK;
}

static inline xccl_status_t
xccl_hier_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    xccl_hier_context_t *ctx = ucs_derived_of(team->ctx, xccl_hier_context_t);
    xccl_seq_schedule_t        *schedule;
    xccl_hier_alltoall_spec_t  spec;
    ucs_memory_type_t          mem_type;
    xccl_status_t              status;

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer,
                                     &mem_type);
    if (status != XCCL_OK) {
        xccl_hier_error("Memtype detection error");
        return XCCL_ERR_INVALID_PARAM;
    }

    spec.pairs.flat = XCCL_HIER_PAIR_FLAT_UCX;
    switch (mem_type) {
        case UCS_MEMORY_TYPE_HOST:
            spec.pairs.node = XCCL_HIER_PAIR_NODE_UCX;
            break;
        case UCS_MEMORY_TYPE_CUDA:
            if (ctx->tls[ucs_ilog2(XCCL_TL_NCCL)].enabled) {
                spec.pairs.node = XCCL_HIER_PAIR_NODE_NCCL;
            } else {
                spec.pairs.node = XCCL_HIER_PAIR_NODE_UCX;
            }
            break;
        default:
            xccl_hier_error("Memory type (%d) is not supported", mem_type);
    }

    build_alltoall_task_schedule(ucs_derived_of(team, xccl_hier_team_t), (*coll_args),
                                 spec, &schedule);
    schedule->req.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->req;
    return XCCL_OK;

}
static inline xccl_status_t
xccl_hier_allreduce_init(xccl_coll_op_args_t *coll_args,
                         xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_seq_schedule_t *schedule;
    xccl_hier_context_t *ctx = ucs_derived_of(team->ctx, xccl_hier_context_t);
    xccl_hier_allreduce_spec_t spec = {
        .pairs              = {
            .node_leaders   = ctx->tls[ucs_ilog2(XCCL_TL_SHARP)].enabled ?
                              XCCL_HIER_PAIR_NODE_LEADERS_SHARP :
                              XCCL_HIER_PAIR_NODE_LEADERS_UCX,
            .socket         = ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled ?
                              XCCL_HIER_PAIR_SOCKET_SHMSEG :
                              XCCL_HIER_PAIR_SOCKET_UCX,
            .socket_leaders = ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled ?
                              XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG :
                              XCCL_HIER_PAIR_SOCKET_LEADERS_UCX,
            .node           = XCCL_HIER_PAIR_NODE_UCX,
        },
    };
    build_allreduce_task_schedule(ucs_derived_of(team, xccl_hier_team_t), (*coll_args),
                                  spec, &schedule);
    schedule->req.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->req;
    return XCCL_OK;
}


static inline xccl_status_t
xccl_hier_bcast_init(xccl_coll_op_args_t *coll_args,
                     xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    xccl_seq_schedule_t *schedule;
    xccl_hier_context_t *ctx = ucs_derived_of(team->ctx, xccl_hier_context_t);

    xccl_hier_bcast_spec_t spec = {
        .use_sm_fanout_get  = 0,
        .pairs              = {
            .node_leaders   = ctx->tls[ucs_ilog2(XCCL_TL_HMC)].enabled ?
                                XCCL_HIER_PAIR_NODE_LEADERS_HMC :
                                XCCL_HIER_PAIR_NODE_LEADERS_UCX,
            .socket         = ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled ?
                                XCCL_HIER_PAIR_SOCKET_SHMSEG :
                                XCCL_HIER_PAIR_SOCKET_UCX,
            .socket_leaders = ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled ?
                                XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG :
                                XCCL_HIER_PAIR_SOCKET_LEADERS_UCX,
            .node           = XCCL_HIER_PAIR_NODE_UCX,
        },
    };
    build_bcast_task_schedule(ucs_derived_of(team, xccl_hier_team_t), (*coll_args),
                              spec, &schedule);
    schedule->req.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->req;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_hier_barrier_init(xccl_coll_op_args_t *coll_args,
                       xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
	xccl_seq_schedule_t *schedule;
    xccl_hier_context_t *ctx = ucs_derived_of(team->ctx, xccl_hier_context_t);
    xccl_hier_barrier_spec_t spec = {
        .pairs              = {
            .node_leaders   = ctx->tls[ucs_ilog2(XCCL_TL_SHARP)].enabled ?
                              XCCL_HIER_PAIR_NODE_LEADERS_SHARP :
                              XCCL_HIER_PAIR_NODE_LEADERS_UCX,
            .socket         = ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled ?
                              XCCL_HIER_PAIR_SOCKET_SHMSEG :
                              XCCL_HIER_PAIR_SOCKET_UCX,
            .socket_leaders = ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled ?
                              XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG :
                              XCCL_HIER_PAIR_SOCKET_LEADERS_UCX,
            .node           = XCCL_HIER_PAIR_NODE_UCX,
        },
    };
    build_barrier_task_schedule(ucs_derived_of(team, xccl_hier_team_t), (*coll_args),
                           spec, &schedule);
    schedule->req.lib = &xccl_team_lib_hier.super;
    (*request) = &schedule->req;
    return XCCL_OK;
}

static xccl_status_t
xccl_hier_collective_init(xccl_coll_op_args_t *coll_args,
                          xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    switch (coll_args->coll_type) {
    case XCCL_ALLREDUCE:
        return xccl_hier_allreduce_init(coll_args, request, team);
    case XCCL_BARRIER:
        return xccl_hier_barrier_init(coll_args, request, team);
    case XCCL_BCAST:
        return xccl_hier_bcast_init(coll_args, request, team);
    case XCCL_ALLTOALL:
        return xccl_hier_alltoall_init(coll_args, request, team);
    }
    return XCCL_ERR_INVALID_PARAM;
}

static xccl_status_t xccl_hier_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_seq_schedule_t *schedule = ucs_container_of(request, xccl_seq_schedule_t, req);
    schedule->tasks[schedule->dep].super.state = XCCL_INPROGRESS;
    xccl_schedule_start(&schedule->super);
    return XCCL_OK;
}

static xccl_status_t xccl_hier_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_seq_schedule_t *schedule = ucs_container_of(request, xccl_seq_schedule_t, req);
    return schedule->super.super.state == XCCL_TASK_STATE_COMPLETED ? XCCL_OK :
        XCCL_INPROGRESS;
}

static xccl_status_t xccl_hier_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_status_t status = xccl_hier_collective_test(request);
    xccl_seq_schedule_t *schedule = ucs_container_of(request, xccl_seq_schedule_t, req);
    xccl_context_t *ctx = schedule->super.tl_ctx->ctx;
    while (XCCL_OK != status) {
        xccl_context_progress(ctx);
        status = xccl_hier_collective_test(request);
    }
    return XCCL_OK;
}

xccl_status_t xccl_hier_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_seq_schedule_t *schedule = ucs_container_of(request, xccl_seq_schedule_t, req);
    free(schedule->tasks);
    free(schedule);
    return XCCL_OK;
}

xccl_team_lib_hier_t xccl_team_lib_hier = {
    .super.name                  = "hier",
    .super.id                    = XCCL_TL_HIER,
    .super.priority              = 150,
    .super.team_lib_config       = {
        .name                    = "HIER tl",
        .prefix                  = "TEAM_HIER_",
        .table                   = xccl_team_lib_hier_config_table,
        .size                    = sizeof(xccl_team_lib_hier_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "HIER tl context",
        .prefix                  = "TEAM_HIER_",
        .table                   = xccl_tl_hier_context_config_table,
        .size                    = sizeof(xccl_tl_hier_context_config_t),
    },
    .super.params.reproducible   = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,    
    .super.params.thread_mode    = XCCL_THREAD_MODE_SINGLE | 
                                   XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage     = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES,
    .super.params.coll_types     = XCCL_COLL_CAP_BARRIER |
                                   XCCL_COLL_CAP_BCAST |
                                   XCCL_COLL_CAP_ALLREDUCE |
                                   XCCL_COLL_CAP_ALLTOALL,
    .super.mem_types             = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                   UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode       = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create   = xccl_hier_create_context,
    .super.team_context_progress = xccl_hier_context_progress,
    .super.team_context_destroy  = xccl_hier_destroy_context,
    .super.team_create_post      = xccl_hier_team_create_post,
    .super.team_create_test      = xccl_hier_team_create_test,
    .super.team_destroy          = xccl_hier_team_destroy,
    .super.team_lib_open         = xccl_hier_open,
    .super.collective_init       = xccl_hier_collective_init,
    .super.collective_post       = xccl_hier_collective_post,
    .super.collective_wait       = xccl_hier_collective_wait,
    .super.collective_test       = xccl_hier_collective_test,
    .super.collective_finalize   = xccl_hier_collective_finalize,
    .super.global_mem_map_start  = NULL,
    .super.global_mem_map_test   = NULL,
    .super.global_mem_unmap      = NULL,
    .tl_lib                      = NULL,
};
