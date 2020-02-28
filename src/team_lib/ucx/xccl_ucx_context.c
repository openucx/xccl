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

xccl_status_t xccl_ucx_create_context(xccl_team_lib_t *lib, xccl_context_config_t *config,
                                      xccl_tl_context_t **context)
{
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucp_ep_params_t ep_params;
    ucp_config_t *ucp_config;
    ucs_status_t status;
    xccl_team_lib_ucx_context_t *ctx =
        (xccl_team_lib_ucx_context_t *)malloc(sizeof(*ctx));
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);

    status = ucp_config_read("", NULL, &ucp_config);
    assert(UCS_OK == status);

    params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                               UCP_PARAM_FIELD_REQUEST_SIZE |
                               UCP_PARAM_FIELD_REQUEST_INIT |
                               UCP_PARAM_FIELD_REQUEST_CLEANUP |
                               UCP_PARAM_FIELD_TAG_SENDER_MASK;
    params.features          = UCP_FEATURE_TAG;
    params.request_size      = sizeof(xccl_ucx_request_t);
    params.request_init      = xccl_ucx_req_init;
    params.request_cleanup   = xccl_ucx_req_cleanup;
    params.tag_sender_mask   = TEAM_UCX_TAG_SENDER_MASK;
    if (config->oob.size > 0) {
        params.field_mask |= UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
        params.estimated_num_eps = config->oob.size;
    }
    
    status = ucp_init(&params, ucp_config, &ctx->ucp_context);
    ucp_config_release(ucp_config);
    assert(UCS_OK == status);

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(ctx->ucp_context, &worker_params,
                               &ctx->ucp_worker);
    assert(UCS_OK == status);

    status = ucp_worker_get_address(ctx->ucp_worker, &ctx->worker_address,
                                    &ctx->ucp_addrlen);
    assert(UCS_OK == status);
    if (config->oob.size > 0) {
        ctx->ucp_eps = (ucp_ep_h*)calloc(config->oob.size, sizeof(ucp_ep_h));
    } else {
        ctx->ucp_eps = NULL;
    }
    ctx->num_to_probe = 10;
    ctx->next_cid     = 0;
    *context = &ctx->super;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_destroy_context(xccl_tl_context_t *team_context)
{
    xccl_team_lib_ucx_context_t *ctx = xccl_derived_of(team_context, xccl_team_lib_ucx_context_t);
    xccl_oob_collectives_t      *oob = &team_context->cfg->oob;
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
