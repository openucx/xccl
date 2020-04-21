/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_ucx_lib.h"
#include "xccl_ucx_team.h"
#include "xccl_ucx_context.h"
#include "allreduce/allreduce.h"
#include "reduce/reduce.h"
#include "fanout/fanout.h"
#include "fanin/fanin.h"
#include "bcast/bcast.h"
#include "barrier/barrier.h"
#include "utils/mem_component.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t xccl_team_lib_ucx_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(xccl_team_lib_ucx_config_t, super),
        UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t xccl_tl_ucx_context_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(xccl_tl_ucx_context_config_t, super),
        UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {"NET_DEVICES", "all",
     "Specifies which network device(s) to use",
     ucs_offsetof(xccl_tl_ucx_context_config_t, devices), UCS_CONFIG_TYPE_STRING_ARRAY
    },

    {NULL}
};

static inline xccl_status_t
xccl_ucx_coll_base_init(xccl_coll_op_args_t *coll_args, xccl_tl_team_t *team,
                        xccl_ucx_collreq_t **request)
{
    xccl_status_t     status;
    ucs_memory_type_t mem_type;

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer,
                                     &mem_type);
    if (status != XCCL_OK) {
        xccl_ucx_error("Memtype detection error");
        return XCCL_ERR_INVALID_PARAM;
    }
    xccl_ucx_info("src_buffer memory type: %s", ucs_memory_type_names[mem_type]);

    //todo malloc ->mpool
    xccl_ucx_collreq_t *req = (xccl_ucx_collreq_t *)malloc(sizeof(*req));
    memcpy(&req->args, coll_args, sizeof(*coll_args));
    req->complete  = XCCL_INPROGRESS;
    req->team      = team;
    req->super.lib = &xccl_team_lib_ucx.super;
    req->tag       = ((xccl_ucx_team_t*)team)->seq_num++;
    req->mem_type  = mem_type;

    (*request)     = req;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_ucx_allreduce_init(xccl_coll_op_args_t *coll_args,
                        xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_ucx_collreq_t *req;
    xccl_ucx_coll_base_init(coll_args, team, &req);
    req->start = xccl_ucx_allreduce_knomial_start;
    (*request) = (xccl_tl_coll_req_t*)&req->super;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_ucx_reduce_init(xccl_coll_op_args_t *coll_args,
                     xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_ucx_collreq_t *req;
    xccl_ucx_coll_base_init(coll_args, team, &req);
    req->start = xccl_ucx_reduce_linear_start;
    (*request) = (xccl_tl_coll_req_t*)&req->super;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_ucx_fanin_init(xccl_coll_op_args_t *coll_args,
                    xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_ucx_collreq_t *req;
    xccl_ucx_coll_base_init(coll_args, team, &req);
    req->start = xccl_ucx_fanin_linear_start;
    (*request) = (xccl_tl_coll_req_t*)&req->super;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_ucx_fanout_init(xccl_coll_op_args_t *coll_args,
                     xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_ucx_collreq_t *req;
    xccl_ucx_coll_base_init(coll_args, team, &req);
    req->start = xccl_ucx_fanout_linear_start;
    (*request) = (xccl_tl_coll_req_t*)&req->super;
    return XCCL_OK;
}

static inline xccl_status_t
xccl_ucx_bcast_init(xccl_coll_op_args_t *coll_args,
                    xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_ucx_collreq_t *req;
    xccl_status_t status = XCCL_OK;
    xccl_ucx_coll_base_init(coll_args, team, &req);
    if (!coll_args->alg.set_by_user) {
        /* Automatic algorithm selection - take knomial */
        req->start = xccl_ucx_bcast_knomial_start;
    } else {
        switch (coll_args->alg.id) {
        case 0:
            req->start = xccl_ucx_bcast_linear_start;
            break;
        case 1:
            req->start = xccl_ucx_bcast_knomial_start;
            break;
        default:
            free(req);
            req = NULL;
            status = XCCL_ERR_INVALID_PARAM;
        }
    }
    (*request) = (xccl_tl_coll_req_t*)&req->super;
    return status;
}

static inline xccl_status_t
xccl_ucx_barrier_init(xccl_coll_op_args_t *coll_args,
                      xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    //TODO alg selection for allreduce shoud happen here
    xccl_ucx_collreq_t *req;
    xccl_ucx_coll_base_init(coll_args, team, &req);
    req->start = xccl_ucx_barrier_knomial_start;
    (*request) = (xccl_tl_coll_req_t*)&req->super;
    return XCCL_OK;
}

static xccl_status_t
xccl_ucx_collective_init(xccl_coll_op_args_t *coll_args,
                         xccl_tl_coll_req_t **request, xccl_tl_team_t *team)
{
    switch (coll_args->coll_type) {
    case XCCL_ALLREDUCE:
        return xccl_ucx_allreduce_init(coll_args, request, team);
    case XCCL_BARRIER:
        return xccl_ucx_barrier_init(coll_args, request, team);
    case XCCL_REDUCE:
        return xccl_ucx_reduce_init(coll_args, request, team);
    case XCCL_BCAST:
        return xccl_ucx_bcast_init(coll_args, request, team);
    case XCCL_FANIN:
        return xccl_ucx_fanin_init(coll_args, request, team);
    case XCCL_FANOUT:
        return xccl_ucx_fanout_init(coll_args, request, team);
    }
    return XCCL_ERR_INVALID_PARAM;
}

static xccl_status_t xccl_ucx_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_ucx_collreq_t *req = ucs_derived_of(request, xccl_ucx_collreq_t);
    return req->start(req);
}

static xccl_status_t xccl_ucx_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_ucx_collreq_t *req = ucs_derived_of(request, xccl_ucx_collreq_t);
    xccl_status_t status;
    while (XCCL_INPROGRESS == req->complete) {
        if (XCCL_OK != (status = req->progress(req))) {
            return status;
        };
    }
    assert(XCCL_OK == req->complete);
    return XCCL_OK;
}

static xccl_status_t xccl_ucx_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_ucx_collreq_t *req = ucs_derived_of(request, xccl_ucx_collreq_t);
    xccl_status_t status;
    if (XCCL_INPROGRESS == req->complete) {
        if (XCCL_OK != (status = req->progress(req))) {
            return status;
        };
    }
    return req->complete;
}

static xccl_status_t xccl_ucx_collective_finalize(xccl_tl_coll_req_t *request)
{
    free(request);
    return XCCL_OK;
}

static xccl_status_t xccl_ucx_lib_open(xccl_team_lib_h self,
                                       xccl_team_lib_config_t *config) {
    xccl_team_lib_ucx_t        *tl  = ucs_derived_of(self, xccl_team_lib_ucx_t);
    xccl_team_lib_ucx_config_t *cfg = ucs_derived_of(config, xccl_team_lib_ucx_config_t);
    
    tl->log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->log_component.name, "%s", "TEAM_UCX");
    xccl_ucx_debug("Team UCX opened");
    if (cfg->super.priority == -1) {
        tl->super.priority = 10;
    } else {
        tl->super.priority = cfg->super.priority;
    }

    return XCCL_OK;
}

static xccl_status_t xccl_ucx_lib_query(xccl_team_lib_h lib, xccl_tl_attr_t *tl_attr) {
    int    num_ibv_devices;
    struct ibv_device **device_list;
    char   (*devices)[16];
    int    i, rc, p;

    if (tl_attr->field_mask & XCCL_TL_ATRR_FIELD_CONTEXT_CREATE_MODE) {
        tl_attr->context_create_mode = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL;
    }

    return XCCL_OK;
}

xccl_team_lib_ucx_t xccl_team_lib_ucx = {
    .super.name                  = "ucx",
    .super.id                    = XCCL_TL_UCX,
    .super.priority              = 10,
    .super.team_lib_config       = {
        .name                    = "UCX team library",
        .prefix                  = "TEAM_UCX_",
        .table                   = xccl_team_lib_ucx_config_table,
        .size                    = sizeof(xccl_team_lib_ucx_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "UCX tl context",
        .prefix                  = "TEAM_UCX_",
        .table                   = xccl_tl_ucx_context_config_table,
        .size                    = sizeof(xccl_tl_ucx_context_config_t),
    },
    .super.team_lib_open         = xccl_ucx_lib_open,
    .super.team_lib_query        = xccl_ucx_lib_query,
    .super.params.reproducible   = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode    = XCCL_THREAD_MODE_SINGLE |
                                   XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage     = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES,
    .super.params.coll_types     = XCCL_COLL_CAP_BARRIER | XCCL_COLL_CAP_FANIN |
                                   XCCL_COLL_CAP_FANOUT | XCCL_COLL_CAP_REDUCE |
                                   XCCL_COLL_CAP_BCAST | XCCL_COLL_CAP_ALLREDUCE,
    .super.ctx_create_mode       = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create   = xccl_ucx_create_context,
    .super.team_context_progress = NULL,
    .super.team_context_destroy  = xccl_ucx_destroy_context,
    .super.team_create_post      = xccl_ucx_team_create_post,
    .super.team_create_test      = xccl_ucx_team_create_test,
    .super.team_destroy          = xccl_ucx_team_destroy,
    .super.collective_init       = xccl_ucx_collective_init,
    .super.collective_post       = xccl_ucx_collective_post,
    .super.collective_wait       = xccl_ucx_collective_wait,
    .super.collective_test       = xccl_ucx_collective_test,
    .super.collective_finalize   = xccl_ucx_collective_finalize,
    .super.global_mem_map_start  = NULL,
    .super.global_mem_map_test   = NULL,
    .super.global_mem_unmap      = NULL,
};

void xccl_ucx_send_completion_cb(void* request, ucs_status_t status)
{
    xccl_ucx_request_t *req = request;
    req->status = XCCL_UCX_REQUEST_DONE;
}

void xccl_ucx_recv_completion_cb(void* request, ucs_status_t status,
                                     ucp_tag_recv_info_t *info)
{
    xccl_ucx_request_t *req = request;
    req->status = XCCL_UCX_REQUEST_DONE;
}
