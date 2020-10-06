/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"

#include "xccl_mrail_lib.h"
#include <ucs/memory/memory_type.h>
#include <pthread.h>

static const char *xccl_tl_names[] = {
    "",
    "ucx",
    "hier",
    "sharp",
    "vmc",
    "shmseg",
    "mrail",
};

static ucs_config_field_t xccl_team_lib_mrail_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(xccl_team_lib_mrail_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {"REPLICATE_TEAM", "hier",
     "Multirail team internal library\n",
     ucs_offsetof(xccl_team_lib_mrail_config_t, replicated_tl_id),
     UCS_CONFIG_TYPE_ENUM(xccl_tl_names)
    },

    {"REPLICAS_NUMBER", "2",
     "Replication number\n",
     ucs_offsetof(xccl_team_lib_mrail_config_t, replicas_num),
     UCS_CONFIG_TYPE_UINT
    },

    {"THREADS_NUMBER", "2",
     "Progress threads number\n",
     ucs_offsetof(xccl_team_lib_mrail_config_t, threads_num),
     UCS_CONFIG_TYPE_UINT
    },

    {"ASYNC_POLL_COUNT", "3",
     "Number of poll count for async threads\n",
     ucs_offsetof(xccl_team_lib_mrail_config_t, thread_poll_cnt),
     UCS_CONFIG_TYPE_UINT
     },

    {NULL}
};

static ucs_config_field_t xccl_tl_mrail_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_mrail_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {"NET_DEVICES", "all",
     "Specifies which network device(s) to use",
     ucs_offsetof(xccl_tl_mrail_context_config_t, devices),
     UCS_CONFIG_TYPE_STRING_ARRAY
    },

    {NULL}
};

static void *progress_context_async(void *progress_thread) {
    xccl_mrail_progress_thread_t  *t = progress_thread;
    xccl_mrail_progress_request_t *req;
    int                           i;

    while(1) {
        pthread_mutex_lock(&t->mutex);

        while(ucs_list_is_empty(&t->list)) {
            pthread_cond_wait(&t->cond, &t->mutex);
            if (t->close) {
                return NULL;
            }
        }

        i = 0;
        do {
            ucs_list_for_each(req, &t->list, list) {
                xccl_context_progress(req->ctx);
                req->completed = xccl_collective_test(req->req);
                if (req->completed == XCCL_OK) {
                    ucs_list_del(&req->list);
                }
            }
            i++;
        } while((i < t->poll_cnt) && (!ucs_list_is_empty(&t->list)));

        pthread_mutex_unlock(&t->mutex);
    }

    return NULL;
}


static xccl_status_t xccl_mrail_open(xccl_team_lib_h self,
                                     xccl_team_lib_config_t *config)
{
    xccl_team_lib_mrail_t        *tl  = ucs_derived_of(self, xccl_team_lib_mrail_t);
    xccl_team_lib_mrail_config_t *cfg = ucs_derived_of(config, xccl_team_lib_mrail_config_t);
    pthread_attr_t               attr;
    int                          i;
    int                          rc;

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", tl->super.name);
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }
    tl->config.replicated_tl_id = UCS_BIT(cfg->replicated_tl_id - 1);
    tl->config.replicas_num     = cfg->replicas_num;
    tl->config.threads_num      = cfg->threads_num;

    if (tl->config.threads_num > 0) {
        /* No need to progress from main thread */
        tl->super.team_context_progress = NULL;
    }

    for(i = 0; i < tl->config.threads_num; i++) {
        pthread_mutex_init(&tl->threads[i].mutex, NULL);
        pthread_cond_init (&tl->threads[i].cond,  NULL);
        ucs_list_head_init(&tl->threads[i].list);
        tl->threads[i].close    = 0;
        tl->threads[i].poll_cnt = cfg->thread_poll_cnt;

        pthread_attr_init(&attr);
        rc = pthread_create(&tl->threads[i].tid, &attr, progress_context_async,
                            &tl->threads[i]);
        if (rc != 0) {
            xccl_mrail_error("failed to spawn thread (%d)", rc);
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    xccl_mrail_debug("team opened");

    return XCCL_OK;
}

static void xccl_mrail_close(xccl_team_lib_h self)
{
    xccl_team_lib_mrail_t *tl  = ucs_derived_of(self, xccl_team_lib_mrail_t);
    int i;

    for(i = 0; i < tl->config.threads_num; i++) {
        tl->threads[i].close = 1;
        pthread_cond_signal(&tl->threads[i].cond);
        pthread_join(tl->threads[i].tid, NULL);
        pthread_mutex_destroy(&tl->threads[i].mutex);
        pthread_cond_destroy(&tl->threads[i].cond);
    }
}

static xccl_status_t xccl_mrail_init_tl(xccl_lib_h *lib_p)
{
    xccl_status_t status;
    xccl_lib_params_t params = {
        .field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE,
        .team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                      XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    };

    status = xccl_lib_init(&params, NULL, lib_p);

    return status;
}

static xccl_status_t xccl_mrail_init_tl_context(xccl_lib_h lib,
                                                xccl_context_h *ctx,
                                                xccl_tl_id_t tl_id,
                                                xccl_oob_collectives_t oob,
                                                const char *devices)
{

    xccl_status_t status;

    xccl_context_params_t params = {
        .field_mask      = XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE |
                           XCCL_CONTEXT_PARAM_FIELD_OOB |
                           XCCL_CONTEXT_PARAM_FIELD_TLS,
        .thread_mode     = XCCL_THREAD_MODE_SINGLE,
        .oob             = oob,
        .tls             = tl_id
    };

    xccl_context_config_t *config;
    xccl_context_config_read(lib, NULL, NULL, &config);
    xccl_context_config_modify(&tl_id, config, "NET_DEVICES", devices);

    status = xccl_context_create(lib, &params, config, ctx);
    xccl_context_config_release(config);

    return status;
}

static xccl_status_t xccl_mrail_init_team(xccl_team_h *team, xccl_context_h ctx,
                                          xccl_team_params_t *params)
{
    xccl_status_t status;

    status = xccl_team_create_post(ctx, params, team);
    return status;
}

xccl_status_t
xccl_mrail_context_create(xccl_team_lib_t *lib, xccl_context_params_t *params,
                          xccl_tl_context_config_t *config,
                          xccl_tl_context_t **context)
{
    xccl_team_lib_mrail_t *mrail = ucs_derived_of(lib, xccl_team_lib_mrail_t);
    xccl_tl_mrail_context_config_t *cfg;
    xccl_mrail_context_t           *ctx;
    xccl_status_t                  status;
    int                            i;
    int                            n_devs;

    xccl_mrail_debug("context create");
    cfg = ucs_derived_of(config, xccl_tl_mrail_context_config_t);
    n_devs = cfg->devices.count;
    if (mrail->config.replicated_tl_id == XCCL_TL_MRAIL) {
        xccl_mrail_warn("team %s is not supported",
                         xccl_tl_names[mrail->config.replicated_tl_id]);
        return XCCL_ERR_INVALID_PARAM;
    }

    ctx = (xccl_mrail_context_t*)malloc(sizeof(*ctx));
    if (ctx == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    memset(ctx->tls, 0, sizeof(ctx->tls));
    status = xccl_mrail_init_tl(&ctx->tl);
    if (status != XCCL_OK) {
        goto free_ctx;
    }

    for(ctx->n_tls = 0; ctx->n_tls < mrail->config.replicas_num; ctx->n_tls++) {
        status = xccl_mrail_init_tl_context(ctx->tl,
                                            &ctx->tls[ctx->n_tls],
                                            mrail->config.replicated_tl_id,
                                            params->oob,
                                            cfg->devices.names[ctx->n_tls % n_devs]);
        if (status != XCCL_OK) {
            xccl_mrail_debug("Failed to init internal contexts");
            goto free_tls;
        }
    }

    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    *context = &ctx->super;

    return XCCL_OK;

free_tls:
    for(ctx->n_tls = ctx->n_tls - 1; ctx->n_tls >= 0; ctx->n_tls--) {
        xccl_context_destroy(ctx->tls[ctx->n_tls]);
    }
    xccl_lib_cleanup(ctx->tl);

free_ctx:
    free(ctx);
    return status;
}

xccl_status_t xccl_mrail_context_destroy(xccl_tl_context_t *team_context)
{
    xccl_mrail_context_t *ctx = ucs_derived_of(team_context, xccl_mrail_context_t);
    int                  i;

    xccl_mrail_debug("context destroy");
    for(i = 0; i < ctx->n_tls; i++) {
        xccl_context_destroy(ctx->tls[i]);
    }
    xccl_lib_cleanup(ctx->tl);

    free(ctx);
    return XCCL_OK;
}

xccl_status_t
xccl_mrail_team_create_post(xccl_tl_context_t *context, xccl_team_params_t *params,
                            xccl_tl_team_t **tl_team)
{
    xccl_mrail_context_t *ctx = ucs_derived_of(context, xccl_mrail_context_t);
    xccl_mrail_team_t    *team;
    xccl_status_t        status;

    team = (xccl_mrail_team_t*)calloc(1, sizeof(xccl_mrail_team_t));
    if (team == NULL){
        return XCCL_ERR_NO_MEMORY;
    }

    memset(team->teams, 0, sizeof(team->teams));
    for(team->n_teams = 0; team->n_teams < ctx->n_tls; team->n_teams++) {
        status = xccl_mrail_init_team(&team->teams[team->n_teams],
                                      ctx->tls[team->n_teams], params);
        if (status != XCCL_OK) {
            xccl_mrail_debug("Failed to init internal teams");
            goto free_teams;
        }
    }

    XCCL_TEAM_SUPER_INIT(team->super, context, params);

    *tl_team = &team->super;
    return XCCL_OK;
free_teams:
/* TODO: destroy teams perhaps shouldn't be there because teams 
 *       are not ready yet
 *   team->n_teams -= 1;
 *   for(; team->n_teams >= 0; team->n_teams--) {
 *       xccl_team_destroy(team->teams[team->n_teams]);
 *   }
 */
    free(team);
    return status;
}

xccl_status_t xccl_mrail_team_create_test(xccl_tl_team_t *tl_team)
{
    xccl_mrail_team_t *team = ucs_derived_of(tl_team, xccl_mrail_team_t);
    xccl_status_t     status;
    int               i;

    for(i = 0; i < team->n_teams; i++) {
        status = xccl_team_create_test(team->teams[i]);
        if (status == XCCL_INPROGRESS) {
            return status;
        }

        if (status != XCCL_OK) {
            xccl_mrail_error("internal team create failed");
            goto free_teams;
        }
    }

    return XCCL_OK;
free_teams:
    team->n_teams -= 1;
    for(; team->n_teams >= 0; team->n_teams--) {
        xccl_team_destroy(team->teams[team->n_teams]);
    }
    free(team);
    return status;
}

xccl_status_t xccl_mrail_team_destroy(xccl_tl_team_t *team)
{
    xccl_mrail_team_t *mrail_team = ucs_derived_of(team, xccl_mrail_team_t);
    int               i;

    for(i = 0; i < mrail_team->n_teams; i++) {
        xccl_team_destroy(mrail_team->teams[i]);
    }

    free(mrail_team);

    return XCCL_OK;
}

xccl_status_t xccl_mrail_context_progress(xccl_tl_context_t *team_context)
{
    int                  i;
    xccl_mrail_context_t  *mrail_ctx = ucs_derived_of(team_context,
                                                     xccl_mrail_context_t);
    xccl_team_lib_mrail_t *mrail     = ucs_derived_of(team_context->lib,
                                                     xccl_team_lib_mrail_t);

    for(i = 0; i < mrail_ctx->n_tls; i++) {
        xccl_context_progress(mrail_ctx->tls[i]);
    }

    return XCCL_OK;
}

static xccl_status_t
xccl_mrail_collective_init(xccl_coll_op_args_t *coll_args,
                           xccl_tl_coll_req_t **request, xccl_tl_team_t *tl_team)
{
    xccl_mrail_coll_req_t *req;
    xccl_mrail_team_t     *team  = ucs_derived_of(tl_team, xccl_mrail_team_t);
    xccl_mrail_context_t  *ctx   = ucs_derived_of(tl_team->ctx, xccl_mrail_context_t);
    xccl_team_lib_mrail_t *mrail = ucs_derived_of(ctx->super.lib, xccl_team_lib_mrail_t);
    xccl_status_t         status;
    xccl_coll_op_args_t   team_args;
    int                   n_reqs;
    int                   buf_len;
    int                   last_buf_len;
    int                   len;
    int                   offset;
    int                   i;
    int                   thread_id;

    req = (xccl_mrail_coll_req_t*)malloc(sizeof(xccl_mrail_coll_req_t));

    if (req == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    if (coll_args->buffer_info.len < team->n_teams) {
        n_reqs  = 1;
        buf_len = 0;
    } else {
        n_reqs  = team->n_teams;
        buf_len = coll_args->buffer_info.len / n_reqs;
    }
    last_buf_len = coll_args->buffer_info.len - (n_reqs - 1)*buf_len;

    memset(req->reqs, 0, n_reqs * sizeof(xccl_coll_req_h));
    team_args = *coll_args;

    for(req->n_reqs = 0; req->n_reqs < n_reqs; req->n_reqs++) {
        len    = buf_len;
        offset = len*req->n_reqs;
        if (req->n_reqs == n_reqs - 1) {
            len = last_buf_len;
        }
        team_args.buffer_info.dst_buffer = (void*)((ptrdiff_t)
                                           coll_args->buffer_info.dst_buffer +
                                           offset);
        team_args.buffer_info.src_buffer = (void*)((ptrdiff_t)
                                           coll_args->buffer_info.src_buffer +
                                           offset);
        team_args.buffer_info.len        = len;
        team_args.reduce_info.count      = len/xccl_dt_size(coll_args->reduce_info.dt);
        status = xccl_collective_init(&team_args, &req->reqs[req->n_reqs].req,
                                      team->teams[req->n_reqs]);
        req->reqs[req->n_reqs].completed = XCCL_INPROGRESS;
        req->reqs[req->n_reqs].ctx       = ctx->tls[req->n_reqs];
 
        if (status != XCCL_OK) {
            xccl_mrail_error("Failed to init collective");
            goto free_req;
        }
    }

    req->super.lib = &xccl_team_lib_mrail.super;
    (*request) = &req->super;
    return XCCL_OK;
free_req:
    free(req);
    return XCCL_ERR_NO_MESSAGE;
}

static xccl_status_t xccl_mrail_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_mrail_coll_req_t  *req   = ucs_derived_of(request, xccl_mrail_coll_req_t);
    xccl_team_lib_mrail_t  *mrail = ucs_derived_of(request->lib, xccl_team_lib_mrail_t);
    xccl_status_t          status;
    int                    i;
    int                    thread_id;

    for(i = 0; i < req->n_reqs; i++) {
        if (mrail->config.threads_num != 0) {
            thread_id = i % mrail->config.threads_num;
            pthread_mutex_lock(&mrail->threads[thread_id].mutex);
        }

        status = xccl_collective_post(req->reqs[i].req);

        if (mrail->config.threads_num != 0) {
            thread_id = i % mrail->config.threads_num;
            ucs_list_add_tail(&mrail->threads[thread_id].list, &req->reqs[i].list);
            pthread_mutex_unlock(&mrail->threads[thread_id].mutex);
        }

        if (status != XCCL_OK) {
            xccl_mrail_error("Failed to post collective");
            return status;
        }
    }

    if (mrail->config.threads_num != 0) {
        for(i = 0; i < req->n_reqs; i++) {
            thread_id = i % mrail->config.threads_num;
            pthread_cond_signal(&mrail->threads[thread_id].cond);
        }
    }

    return XCCL_OK;
}

static xccl_status_t xccl_mrail_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_mrail_coll_req_t *req   = ucs_derived_of(request, xccl_mrail_coll_req_t);
    xccl_team_lib_mrail_t *mrail = ucs_derived_of(request->lib, xccl_team_lib_mrail_t);
    xccl_status_t         status;
    int                   i;
    xccl_status_t         global_status;

    global_status = XCCL_OK;
    for(i = 0; i < req->n_reqs; i++) {
        if (mrail->config.threads_num != 0) {
            status = req->reqs[i].completed;
        } else {
            status = xccl_collective_test(req->reqs[i].req);
        }

        if ((status != XCCL_OK) && (status != XCCL_INPROGRESS)) {
            xccl_mrail_error("Error occured during collective test");
            return status;
        }

        if (status == XCCL_INPROGRESS) {
            global_status = XCCL_INPROGRESS;
        }
    }

    return global_status;
}

static xccl_status_t xccl_mrail_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_status_t status;

    do {
        status = xccl_mrail_collective_test(request);
    } while (status == XCCL_INPROGRESS);

    return XCCL_OK;
}

xccl_status_t xccl_mrail_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_mrail_coll_req_t *req = ucs_derived_of(request, xccl_mrail_coll_req_t);
    int                   i;
    xccl_status_t         status;
    xccl_status_t         global_status;

    global_status = XCCL_OK;
    for(i = 0; i < req->n_reqs; i++) {
        status = xccl_collective_finalize(req->reqs[i].req);
        if (status != XCCL_OK) {
            global_status = status;
            xccl_error("collective finalize error");
        }
    }
    free(request);

    return global_status;
}

xccl_team_lib_mrail_t xccl_team_lib_mrail = {
    .super.name                  = "mrail",
    .super.id                    = XCCL_TL_MRAIL,
    .super.priority              = 100,
    .super.team_lib_config       = {
        .name                    = "Multirail team library",
        .prefix                  = "TEAM_MRAIL_",
        .table                   = xccl_team_lib_mrail_config_table,
        .size                    = sizeof(xccl_team_lib_mrail_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "Multirail tl context",
        .prefix                  = "TEAM_MRAIL_",
        .table                   = xccl_tl_mrail_context_config_table,
        .size                    = sizeof(xccl_tl_mrail_context_config_t)
    },
    .super.params.reproducible   = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode    = XCCL_THREAD_MODE_SINGLE  |
                                   XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage     = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES,
    .super.params.coll_types     = XCCL_COLL_CAP_BARRIER   |
                                   XCCL_COLL_CAP_BCAST     |
                                   XCCL_COLL_CAP_ALLREDUCE,
    .super.mem_types             = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                   UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode       = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create   = xccl_mrail_context_create,
    .super.team_context_destroy  = xccl_mrail_context_destroy,
    .super.team_create_post      = xccl_mrail_team_create_post,
    .super.team_create_test      = xccl_mrail_team_create_test,
    .super.team_destroy          = xccl_mrail_team_destroy,
    .super.team_context_progress = xccl_mrail_context_progress,
    .super.team_lib_open         = xccl_mrail_open,
    .super.team_lib_close        = xccl_mrail_close,
    .super.collective_init       = xccl_mrail_collective_init,
    .super.collective_post       = xccl_mrail_collective_post,
    .super.collective_wait       = xccl_mrail_collective_wait,
    .super.collective_test       = xccl_mrail_collective_test,
    .super.collective_finalize   = xccl_mrail_collective_finalize,
    .super.global_mem_map_start  = NULL,
    .super.global_mem_map_test   = NULL,
    .super.global_mem_unmap      = NULL,
};
