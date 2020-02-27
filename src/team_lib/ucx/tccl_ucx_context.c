/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "tccl_ucx_context.h"
#include "tccl_ucx_tag.h"
#include "tccl_ucx_ep.h"
#include <stdlib.h>

static void tccl_ucx_req_init(void* request)
{
    tccl_ucx_request_t *req = (tccl_ucx_request_t*)request;
    req->status = TCCL_UCX_REQUEST_ACTIVE;
}

static void tccl_ucx_req_cleanup(void* request){ }

tccl_status_t tccl_ucx_create_context(tccl_team_lib_t *lib, tccl_context_config_t *config,
                                      tccl_tl_context_t **context)
{
    ucp_params_t params;
    ucp_worker_params_t worker_params;
    ucp_ep_params_t ep_params;
    ucp_config_t *ucp_config;
    ucs_status_t status;
    tccl_team_lib_ucx_context_t *ctx =
        (tccl_team_lib_ucx_context_t *)malloc(sizeof(*ctx));
    TCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);

    status = ucp_config_read("", NULL, &ucp_config);
    assert(UCS_OK == status);

    params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                               UCP_PARAM_FIELD_REQUEST_SIZE |
                               UCP_PARAM_FIELD_REQUEST_INIT |
                               UCP_PARAM_FIELD_REQUEST_CLEANUP |
                               UCP_PARAM_FIELD_TAG_SENDER_MASK;
    params.features          = UCP_FEATURE_TAG;
    params.request_size      = sizeof(tccl_ucx_request_t);
    params.request_init      = tccl_ucx_req_init;
    params.request_cleanup   = tccl_ucx_req_cleanup;
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
    return TCCL_OK;
}

tccl_status_t tccl_ucx_destroy_context(tccl_tl_context_t *team_context)
{
    tccl_team_lib_ucx_context_t *ctx = tccl_derived_of(team_context, tccl_team_lib_ucx_context_t);
    tccl_oob_collectives_t      *oob = &team_context->cfg->oob;
    void *tmp;

    if (ctx->ucp_eps) {
        close_eps(ctx->ucp_eps, oob->size, ctx->ucp_worker);
        tmp = malloc(oob->size);
        oob->allgather(tmp, tmp, 1, oob->coll_context);
        free(tmp);
        free(ctx->ucp_eps);
    }
    ucp_worker_destroy(ctx->ucp_worker);
    ucp_cleanup(ctx->ucp_context);
    free(ctx);
    return TCCL_OK;
}
