/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include "xccl_ucx_context.h"
#include "xccl_ucx_tag.h"
#include "xccl_ucx_ep.h"
#include <stdlib.h>

static void xccl_ucx_req_init(void* request)
{
    xccl_ucx_request_t *req = (xccl_ucx_request_t*)request;
    req->status = XCCL_UCX_REQUEST_ACTIVE;
}

static void xccl_ucx_req_cleanup(void* request){ }

xccl_status_t xccl_ucx_create_context(xccl_team_lib_t *lib,
                                      xccl_context_params_t *params,
                                      xccl_tl_context_config_t *config,
                                      xccl_tl_context_t **context)
{
    xccl_team_lib_ucx_context_t *ctx  =
        (xccl_team_lib_ucx_context_t *)malloc(sizeof(*ctx));
    xccl_tl_ucx_context_config_t *cfg =
        ucs_derived_of(config, xccl_tl_ucx_context_config_t);
    ucp_params_t        ucp_params;
    ucp_worker_params_t worker_params;
    ucp_ep_params_t     ep_params;
    ucp_config_t        *ucp_config;
    ucs_status_t        status;
    ucp_worker_attr_t   worker_attr;
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);

    status = ucp_config_read(config->env_prefix, NULL, &ucp_config);
    assert(UCS_OK == status);

    ucp_params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                                   UCP_PARAM_FIELD_REQUEST_SIZE |
                                   UCP_PARAM_FIELD_REQUEST_INIT |
                                   UCP_PARAM_FIELD_REQUEST_CLEANUP |
                                   UCP_PARAM_FIELD_TAG_SENDER_MASK |
                                   UCP_PARAM_FIELD_ESTIMATED_NUM_PPN;

    ucp_params.features          = UCP_FEATURE_TAG;
    ucp_params.request_size      = sizeof(xccl_ucx_request_t);
    ucp_params.request_init      = xccl_ucx_req_init;
    ucp_params.request_cleanup   = xccl_ucx_req_cleanup;
    ucp_params.tag_sender_mask   = TEAM_UCX_TAG_SENDER_MASK;
    ucp_params.estimated_num_ppn = cfg->ppn;
    if (params->oob.size > 0) {
        ucp_params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        ucp_params.estimated_num_eps = params->oob.size;
    }

    //TODO: need to fix for case of multiple devices
    ucp_config_modify(ucp_config, "NET_DEVICES", cfg->devices.names[0]);
    status = ucp_init(&ucp_params, ucp_config, &ctx->ucp_context);
    ucp_config_release(ucp_config);
    assert(UCS_OK == status);

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    switch (params->thread_mode) {
        case XCCL_THREAD_MODE_SINGLE:
            worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
            break;
        case XCCL_THREAD_MODE_MULTIPLE:
            worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
            break;
        default:
            xccl_ucx_warn("Incorrect value of context thread mode, using single");
            worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    }
    status = ucp_worker_create(ctx->ucp_context, &worker_params,
                               &ctx->ucp_worker);
    assert(UCS_OK == status);
    if (params->thread_mode == XCCL_THREAD_MODE_MULTIPLE) {
        worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE;
        ucp_worker_query(ctx->ucp_worker, &worker_attr);
        if (worker_attr.thread_mode != UCS_THREAD_MODE_MULTI) {
            xccl_ucx_warn("Thread mode multiple is not supported");
        }
    }

    status = ucp_worker_get_address(ctx->ucp_worker, &ctx->worker_address,
                                    &ctx->ucp_addrlen);
    assert(UCS_OK == status);
    if (params->oob.size > 0) {
        ctx->ucp_eps = (ucp_ep_h*)calloc(params->oob.size, sizeof(ucp_ep_h));
    } else {
        ctx->ucp_eps = NULL;
    }
    ctx->block_stream              = cfg->block_stream;
    ctx->num_to_probe              = cfg->num_to_probe;
    ctx->barrier_kn_radix          = cfg->barrier_kn_radix;
    ctx->bcast_kn_radix            = cfg->bcast_kn_radix;
    ctx->reduce_kn_radix           = cfg->reduce_kn_radix;
    ctx->allreduce_kn_radix        = cfg->allreduce_kn_radix;
    ctx->allreduce_alg_id          = cfg->allreduce_alg_id;
    ctx->alltoall_pairwise_chunk   = cfg->alltoall_pairwise_chunk;
    ctx->alltoall_pairwise_reverse = cfg->alltoall_pairwise_reverse;
    ctx->alltoall_pairwise_barrier = cfg->alltoall_pairwise_barrier;

    ctx->next_cid           = 0;
    *context = &ctx->super;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_destroy_context(xccl_tl_context_t *team_context)
{
    xccl_team_lib_ucx_context_t *ctx = ucs_derived_of(team_context, xccl_team_lib_ucx_context_t);
    xccl_oob_collectives_t      *oob = &team_context->params.oob;
    void *tmp;

    if (ctx->ucp_eps) {
        close_eps(ctx->ucp_eps, oob->size, ctx->ucp_worker);
        tmp = malloc(oob->size);
        xccl_oob_allgather(tmp, tmp, 1, oob);
        free(tmp);
        free(ctx->ucp_eps);
    }
    ucp_worker_destroy(ctx->ucp_worker);
    ucp_cleanup(ctx->ucp_context);
    free(ctx);
    return XCCL_OK;
}
