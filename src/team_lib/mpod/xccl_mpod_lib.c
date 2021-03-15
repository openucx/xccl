#include "xccl_mpod_lib.h"
#include "mem_component.h"
#include <ucs/memory/memory_type.h>
#include <stdbool.h>
#include <pthread.h>

static ucs_config_field_t xccl_team_lib_mpod_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_team_lib_mpod_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t xccl_tl_mpod_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_mpod_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {NULL}
};

static xccl_status_t xccl_mpod_open(xccl_team_lib_h self,
                                    xccl_team_lib_config_t *config)
{
    xccl_team_lib_mpod_t *tl  = ucs_derived_of(self, xccl_team_lib_mpod_t);
    xccl_team_lib_mpod_config_t *cfg = ucs_derived_of(config, xccl_team_lib_mpod_config_t);

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_MPOD");

    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }

    return XCCL_OK;
}


/*****************************************************************************/
/* context related functions */
/*****************************************************************************/

static xccl_status_t xccl_mpod_create_context(xccl_team_lib_t *lib,
                                              xccl_context_params_t *params,
                                              xccl_tl_context_config_t *config,
                                              xccl_tl_context_t **context)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_context_t *ctx = malloc(sizeof(*ctx));
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);

    ctx->lib.nccl = NULL;
    ctx->lib.ucx = NULL;

    xccl_lib_t *xccl_lib = &xccl_static_lib;
    for (int i = 0; i < xccl_lib->n_libs_opened; i++) {
        if (xccl_lib->libs[i]->id == XCCL_TL_NCCL) {
            ctx->lib.nccl = xccl_lib->libs[i];
        } else if (xccl_lib->libs[i]->id == XCCL_TL_UCX) {
            ctx->lib.ucx = xccl_lib->libs[i];
        }
    }

    /* mpod depends on NCCL and UCX; make sure those libraries have
     * been compiled in */
    assert(ctx->lib.nccl);
    assert(ctx->lib.ucx);

    /* create internal contexts for NCCL and UCX; these are not
     * visible to the user.  We mostly reuse the user-provided
     * parameters, but replace the TLS. */

    /* FIXME: we do not have access to the user-provided context
     * config at this level, so we are rolling back to the default
     * config. */

    /* NCCL context */
    status = ctx->lib.nccl->team_context_create(ctx->lib.nccl, params,
                                               NULL, &ctx->context.nccl);
    xccl_mpod_err_pop(status, fn_fail);

    /* UCX slice context */
    status = ctx->lib.ucx->team_context_create(ctx->lib.ucx, params,
                                               NULL, &ctx->context.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    /* UCX flat context */
    status = ctx->lib.ucx->team_context_create(ctx->lib.ucx, params,
                                               NULL, &ctx->context.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

    *context = &ctx->super;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t xccl_mpod_destroy_context(xccl_tl_context_t *context)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_context_t *ctx = ucs_derived_of(context, xccl_mpod_context_t);

    /* NCCL context */
    status = ctx->lib.nccl->team_context_destroy(ctx->context.nccl);
    xccl_mpod_err_pop(status, fn_fail);

    /* UCX slice context */
    status = ctx->lib.ucx->team_context_destroy(ctx->context.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    /* UCX flat context */
    status = ctx->lib.ucx->team_context_destroy(ctx->context.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

    free(ctx);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}


/*****************************************************************************/
/* Virtualized OOB functions */
/*****************************************************************************/

typedef enum {
    VIRTUAL_OOB_REQ_KIND__NCCL,
    VIRTUAL_OOB_REQ_KIND__UCX,
} virtual_oob_req_kind_e;

typedef struct {
    virtual_oob_req_kind_e kind;

    void *tmpbuf;
    xccl_mpod_team_t *mpod_team;
    void *rbuf;
    size_t msglen;
    void *real_req;
} virtual_oob_req_s;

static xccl_status_t virtual_oob_allgather_nccl(void *sbuf, void *rbuf, size_t msglen,
                                                int rank, xccl_ep_range_t range,
                                                void *coll_context, void **req_p)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_team_t *mpod_team = (xccl_mpod_team_t *) coll_context;
    virtual_oob_req_s *req = (virtual_oob_req_s *) malloc(sizeof(virtual_oob_req_s));

    req->kind = VIRTUAL_OOB_REQ_KIND__NCCL;
    req->tmpbuf = malloc(msglen * mpod_team->user_oob_coll.size);
    req->mpod_team = mpod_team;
    req->rbuf = rbuf;
    req->msglen = msglen;

    status = mpod_team->user_oob_coll.allgather(sbuf, req->tmpbuf, msglen,
                                                rank + mpod_team->pod_id * mpod_team->pod_size,
                                                range, mpod_team->user_oob_coll.coll_context,
                                                &req->real_req);
    if (status != XCCL_INPROGRESS)
        xccl_mpod_err_pop(status, fn_fail);

    *req_p = req;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t virtual_oob_allgather_ucx(void *sbuf, void *rbuf, size_t msglen,
                                               int rank, xccl_ep_range_t range,
                                               void *coll_context, void **req_p)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_team_t *mpod_team = (xccl_mpod_team_t *) coll_context;
    virtual_oob_req_s *req = (virtual_oob_req_s *) malloc(sizeof(virtual_oob_req_s));

    req->kind = VIRTUAL_OOB_REQ_KIND__UCX;
    req->tmpbuf = malloc(msglen * mpod_team->user_oob_coll.size);
    req->mpod_team = mpod_team;
    req->rbuf = rbuf;
    req->msglen = msglen;

    status = mpod_team->user_oob_coll.allgather(sbuf, req->tmpbuf, msglen,
                                                rank * mpod_team->pod_size + mpod_team->slice_id,
                                                range, mpod_team->user_oob_coll.coll_context,
                                                &req->real_req);
    if (status != XCCL_INPROGRESS)
        xccl_mpod_err_pop(status, fn_fail);

    *req_p = req;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t virtual_oob_req_test(void *req_)
{
    xccl_status_t status = XCCL_OK;
    virtual_oob_req_s *req = (virtual_oob_req_s *) req_;
    xccl_mpod_team_t *team = req->mpod_team;

    status = team->user_oob_coll.req_test(req->real_req);
    if (status != XCCL_INPROGRESS)
        xccl_mpod_err_pop(status, fn_fail);

    /* if the request has completed, copy the data to the appropriate
     * user buffers */
    if (status == XCCL_OK) {
        if (req->kind == VIRTUAL_OOB_REQ_KIND__NCCL) {
            memcpy(req->rbuf, (void *) ((uintptr_t) req->tmpbuf + team->pod_id * team->pod_size * req->msglen),
                   req->msglen * team->pod_size);
        } else {
            for (int i = 0; i < team->num_pods; i++) {
                uintptr_t s_offset = (i * team->pod_size + team->slice_id) * req->msglen;
                uintptr_t d_offset = i * req->msglen;
                memcpy((void *) ((uintptr_t) req->rbuf + d_offset),
                       (void *) ((uintptr_t) req->tmpbuf + s_offset), req->msglen);
            }
        }
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t virtual_oob_req_free(void *req_)
{
    xccl_status_t status = XCCL_OK;
    virtual_oob_req_s *req = (virtual_oob_req_s *) req_;
    xccl_mpod_team_t *team = req->mpod_team;

    status = team->user_oob_coll.req_free(req->real_req);
    xccl_mpod_err_pop(status, fn_fail);

    free(req->tmpbuf);
    free(req);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

/*****************************************************************************/
/* Team functions */
/*****************************************************************************/

static xccl_status_t xccl_mpod_team_create_post(xccl_tl_context_t *context,
                                                xccl_team_params_t *params,
                                                xccl_tl_team_t **team)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_context_t *ctx = ucs_derived_of(context, xccl_mpod_context_t);
    xccl_mpod_team_t *mpod_team = malloc(sizeof(*mpod_team));

    XCCL_TEAM_SUPER_INIT(mpod_team->super, context, params);

    /* read user environment to figure out how to split processes into
     * PODs. */
    char *podsize_str = getenv("XCCL_POD_SIZE");
    if (podsize_str == NULL) {
        mpod_team->pod_size = params->oob.size;
        mpod_team->num_pods = 1;
        mpod_team->pod_id = 0;
        mpod_team->slice_id = 0;
    } else {
        mpod_team->pod_size = atoi(podsize_str);
        mpod_team->num_pods = params->oob.size / mpod_team->pod_size;
        mpod_team->pod_id = params->oob.rank / mpod_team->pod_size;
        mpod_team->slice_id = params->oob.rank % mpod_team->pod_size;

        if (params->oob.size % mpod_team->pod_size) {
            /* FIXME: we only support the case where all pods are of the same size */
            xccl_mpod_error("unequal sized pods are not supported");
            status = XCCL_ERR_NO_MESSAGE;
            goto fn_fail;
        }
    }
    mpod_team->context = ctx;
    mpod_team->user_params = *params;
    memcpy(&mpod_team->user_oob_coll, &params->oob, sizeof(xccl_oob_collectives_t));

    /* NCCL team */
    xccl_team_params_t nccl_params = *params;
    nccl_params.oob.allgather = virtual_oob_allgather_nccl;
    nccl_params.oob.req_test = virtual_oob_req_test;
    nccl_params.oob.req_free = virtual_oob_req_free;
    nccl_params.oob.coll_context = (void *) mpod_team;
    nccl_params.oob.rank = params->oob.rank % mpod_team->pod_size;
    nccl_params.oob.size = mpod_team->pod_size;

    status = ctx->lib.nccl->team_create_post(ctx->context.nccl, &nccl_params, &mpod_team->team.nccl);
    xccl_mpod_err_pop(status, fn_fail);

    /* UCX slice team */
    /* FIXME: we support only full range process layouts right now */
    if (!(params->range.type == XCCL_EP_RANGE_FULL ||
          (params->range.type == XCCL_EP_RANGE_STRIDED && params->range.strided.start == 0 &&
           params->range.strided.stride == 1))) {
        xccl_mpod_error("only full ranges are supported");
        status = XCCL_ERR_NO_MESSAGE;
        goto fn_fail;
    }
    xccl_team_params_t ucx_params = *params;
    ucx_params.oob.allgather = virtual_oob_allgather_ucx;
    ucx_params.oob.req_test = virtual_oob_req_test;
    ucx_params.oob.req_free = virtual_oob_req_free;
    ucx_params.oob.coll_context = (void *) mpod_team;
    ucx_params.oob.rank = mpod_team->pod_id;
    ucx_params.oob.size = mpod_team->num_pods;

    status = ctx->lib.ucx->team_create_post(ctx->context.ucx_slice, &ucx_params, &mpod_team->team.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    /* The UCX team implementation does not seem to be able to support
     * simultaneous creation of multiple teams.  This is really a bug
     * in the UCX team code, but we workaround it by serializing the
     * slice and flat team creations. */
    mpod_team->ucx_team_state = UCX_TEAM_STATE__SLICE_TEAM_INITIATED;

    *team = &mpod_team->super;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t xccl_mpod_team_create_test(xccl_tl_team_t *team)
{
    xccl_status_t status = XCCL_OK;
    _Bool is_complete = true;
    xccl_mpod_team_t *mpod_team  = ucs_derived_of(team, xccl_mpod_team_t);

    status = mpod_team->context->lib.nccl->team_create_test(mpod_team->team.nccl);
    if (status == XCCL_INPROGRESS) {
        is_complete = false;
    } else {
        xccl_mpod_err_pop(status, fn_fail);
    }

    if (mpod_team->ucx_team_state == UCX_TEAM_STATE__SLICE_TEAM_INITIATED) {
        status = mpod_team->context->lib.ucx->team_create_test(mpod_team->team.ucx_slice);
        if (status == XCCL_INPROGRESS) {
            is_complete = false;
        } else {
            xccl_mpod_err_pop(status, fn_fail);

            /* slice team is ready; initiate the flat team now */
            status = mpod_team->context->lib.ucx->team_create_post(mpod_team->context->context.ucx_flat,
                                                                   &mpod_team->user_params,
                                                                   &mpod_team->team.ucx_flat);
            xccl_mpod_err_pop(status, fn_fail);

            mpod_team->ucx_team_state = UCX_TEAM_STATE__FLAT_TEAM_INITIATED;
        }
    }

    if (mpod_team->ucx_team_state == UCX_TEAM_STATE__FLAT_TEAM_INITIATED) {
        status = mpod_team->context->lib.ucx->team_create_test(mpod_team->team.ucx_flat);
        if (status == XCCL_INPROGRESS) {
            is_complete = false;
        } else {
            xccl_mpod_err_pop(status, fn_fail);
        }
    }

    if (is_complete) {
        status = XCCL_OK;
    } else {
        status = XCCL_INPROGRESS;
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t xccl_mpod_team_destroy(xccl_tl_team_t *team)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_team_t *mpod_team  = ucs_derived_of(team, xccl_mpod_team_t);

    status = mpod_team->context->lib.nccl->team_destroy(mpod_team->team.nccl);
    xccl_mpod_err_pop(status, fn_fail);

    status = mpod_team->context->lib.ucx->team_destroy(mpod_team->team.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    status = mpod_team->context->lib.ucx->team_destroy(mpod_team->team.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

    free(mpod_team);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}


/*****************************************************************************/
/* Collective functions */
/*****************************************************************************/

static xccl_status_t xccl_mpod_collective_init(xccl_coll_op_args_t *coll_args,
                                               xccl_tl_coll_req_t **request,
                                               xccl_tl_team_t *team)
{
    xccl_mpod_team_t *mpod_team  = ucs_derived_of(team, xccl_mpod_team_t);
    xccl_mpod_coll_req_t *req;
    xccl_status_t status = XCCL_OK;

    req = (xccl_mpod_coll_req_t *) malloc(sizeof(xccl_mpod_coll_req_t));
    req->team = mpod_team;
    memcpy(&req->coll_args, coll_args, sizeof(xccl_coll_op_args_t));

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer, &req->memtype);
    xccl_mpod_err_pop(status, fn_fail);

    if (coll_args->coll_type == XCCL_BARRIER) {
        status = xccl_mpod_barrier_init(req);
        xccl_mpod_err_pop(status, fn_fail);
    } else if (coll_args->coll_type == XCCL_ALLTOALLV) {
        status = xccl_mpod_alltoallv_init(req);
        xccl_mpod_err_pop(status, fn_fail);
    } else if (req->memtype == UCS_MEMORY_TYPE_HOST) {
        status = xccl_mpod_cpu_init(req);
        xccl_mpod_err_pop(status, fn_fail);
    } else {
        switch (coll_args->coll_type) {
        case XCCL_ALLGATHER:
            status = xccl_mpod_allgather_init(req);
            xccl_mpod_err_pop(status, fn_fail);
            break;

        case XCCL_ALLREDUCE:
            status = xccl_mpod_allreduce_init(req);
            xccl_mpod_err_pop(status, fn_fail);
            break;

        case XCCL_BCAST:
            status = xccl_mpod_bcast_init(req);
            xccl_mpod_err_pop(status, fn_fail);
            break;

        case XCCL_ALLTOALL:
            status = xccl_mpod_alltoall_init(req);
            xccl_mpod_err_pop(status, fn_fail);
            break;

        default:
            assert(0);
        }
    }

    req->super.lib = &xccl_team_lib_mpod.super;
    req->self = req;
    (*request) = &req->super;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_mpod_coll_req_t *pending_reqs = NULL;
static pthread_mutex_t req_mutex = PTHREAD_MUTEX_INITIALIZER;
static int req_id = 0;

static xccl_status_t xccl_mpod_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_mpod_coll_req_t *req = ucs_derived_of(request, xccl_mpod_coll_req_t);
    xccl_status_t status = XCCL_OK;

    pthread_mutex_lock(&req_mutex);

    req->req_id = req_id++;

    if (pending_reqs == NULL) {
        status = req->collective_post(req);
        xccl_mpod_err_pop(status, fn_fail);
    }

    HASH_ADD_PTR(pending_reqs, self, req);

  fn_exit:
    pthread_mutex_unlock(&req_mutex);
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t xccl_mpod_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_mpod_coll_req_t *req = ucs_derived_of(request, xccl_mpod_coll_req_t);
    xccl_status_t status = XCCL_OK;

    while (1) {
        status = req->collective_test(req);
        if (status != XCCL_INPROGRESS)
            break;
    }

    return status;
}

static xccl_status_t xccl_mpod_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_mpod_coll_req_t *req = ucs_derived_of(request, xccl_mpod_coll_req_t);
    xccl_status_t status = XCCL_OK;

    pthread_mutex_lock(&req_mutex);

    xccl_mpod_coll_req_t *next = pending_reqs;
    do {
        if (next == NULL)
            break;

        /* test the first request in the queue, and if it has
         * completed, dequeue it and post the next request */
        status = next->collective_test(next);
        xccl_mpod_err_pop(status, fn_fail);

        if (status == XCCL_OK) {
            HASH_DEL(pending_reqs, next);

            next = pending_reqs;
            if (next) {
                status = next->collective_post(next);
                xccl_mpod_err_pop(status, fn_fail);
            }
        }
    } while (status == XCCL_OK);

    /* find the current request in the queue.  if we cannot find the
     * request in the queue, it means that it either just completed or
     * had completed in one of the previous test operations. */
    xccl_mpod_coll_req_t *r;
    HASH_FIND_PTR(pending_reqs, &req->self, r);
    if (r == NULL) {
        status = XCCL_OK;
    } else {
        status = XCCL_INPROGRESS;
    }

  fn_exit:
    pthread_mutex_unlock(&req_mutex);
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t xccl_mpod_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_mpod_coll_req_t *req = ucs_derived_of(request, xccl_mpod_coll_req_t);
    xccl_status_t status = XCCL_OK;

    status = req->collective_finalize(req);
    xccl_mpod_err_pop(status, fn_fail);

    free(req);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_team_lib_mpod_t xccl_team_lib_mpod = {
    .super.name                   = "mpod",
    .super.id                     = XCCL_TL_MPOD,
    .super.priority               = 90,
    .super.team_lib_config        =
    {
        .name                     = "MPOD team library",
        .prefix                   = "TEAM_MPOD_",
        .table                    = xccl_team_lib_mpod_config_table,
        .size                     = sizeof(xccl_team_lib_mpod_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "MPOD tl context",
        .prefix                  = "TEAM_MPOD_",
        .table                   = xccl_tl_mpod_context_config_table,
        .size                    = sizeof(xccl_tl_mpod_context_config_t),
    },
    .super.params.reproducible    = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode     = XCCL_THREAD_MODE_SINGLE | XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage      = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES,
    .super.params.coll_types      = XCCL_COLL_CAP_ALLREDUCE | XCCL_COLL_CAP_BARRIER |
                                    XCCL_COLL_CAP_ALLGATHER | XCCL_COLL_CAP_BCAST |
                                    XCCL_COLL_CAP_ALLTOALL | XCCL_COLL_CAP_ALLTOALLV,
    .super.mem_types              = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode        = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create    = xccl_mpod_create_context,
    .super.team_context_destroy   = xccl_mpod_destroy_context,
    .super.team_context_progress  = NULL,
    .super.team_create_post       = xccl_mpod_team_create_post,
    .super.team_create_test       = xccl_mpod_team_create_test,
    .super.team_destroy           = xccl_mpod_team_destroy,
    .super.team_lib_open          = xccl_mpod_open,
    .super.collective_init        = xccl_mpod_collective_init,
    .super.collective_post        = xccl_mpod_collective_post,
    .super.collective_wait        = xccl_mpod_collective_wait,
    .super.collective_test        = xccl_mpod_collective_test,
    .super.collective_finalize    = xccl_mpod_collective_finalize,
    .super.global_mem_map_start   = NULL,
    .super.global_mem_map_test    = NULL,
    .super.global_mem_unmap       = NULL,
};
