/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include <xccl_context.h>
#include <xccl_progress_queue.h>
#include <xccl_ucs.h>
#include <ucs/sys/math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static xccl_status_t find_tlib_by_name(xccl_lib_t *lib, const char *tlib_name,
                                       int *tl_index) {
    int i;
    for (i=0; i<lib->n_libs_opened; i++) {
        if (0 == strcmp(tlib_name, lib->libs[i]->name)) {
            *tl_index = i;
            return XCCL_OK;
        }
    }
    *tl_index = -1;
    return XCCL_ERR_NO_ELEM;
}

xccl_status_t xccl_context_create(xccl_lib_h lib,
                                  const xccl_context_params_t *params,
                                  const xccl_context_config_t *config,
                                  xccl_context_h *context)
{
    xccl_context_t    *ctx            = malloc(sizeof(xccl_context_t));
    int               num_tls         = ucs_popcount(params->tls);
    uint64_t          default_tls     = XCCL_TL_ALL;
    uint64_t          tls             = 0;
    xccl_context_config_t *dfl_config = NULL;
    uint64_t          i;
    xccl_team_lib_t   *tlib;
    xccl_tl_context_t *tl_ctx;
    int               tl_index;

    ctx->lib = lib;
    memcpy(&ctx->params, params, sizeof(xccl_context_params_t));

    if (params->field_mask & XCCL_CONTEXT_PARAM_FIELD_TLS) {
        tls = params->tls;
    } else {
        tls = default_tls;
    }

    ctx->tl_ctx = (xccl_tl_context_t**)malloc(sizeof(xccl_tl_context_t*)*num_tls);
    ctx->n_tl_ctx = 0;

    if (config == NULL) {
        xccl_context_config_read(lib, NULL, NULL, &dfl_config);
        config = dfl_config;
    }
    /* TODO: use ucs_for_each_bit */
    for (i = 1; i < XCCL_TL_LAST; i = i << 1) {
        if ((i & tls) && find_tlib_by_name(lib, xccl_tl_str(i), &tl_index) == XCCL_OK) {
            tlib = lib->libs[tl_index];
            if (tlib->team_context_create(tlib, &ctx->params, config->configs[tl_index], &tl_ctx) == XCCL_OK) {
                ctx->tl_ctx[ctx->n_tl_ctx++] = tl_ctx;
                xccl_ctx_progress_queue_init(&tl_ctx->pq);
            }
        }
    }

    if (dfl_config != NULL) {
        xccl_context_config_release(dfl_config);
    }

    *context = ctx;
    return XCCL_OK;
}

xccl_status_t xccl_context_progress(xccl_context_h context)
{
    xccl_tl_context_t *tl_ctx;
    xccl_status_t     status;
    int               i;

    for (i = 0; i < context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        if (tl_ctx->lib->team_context_progress) {
            status = tl_ctx->lib->team_context_progress(tl_ctx);
            if (status != XCCL_OK) {
                return status;
            }
            xccl_ctx_progress_queue(tl_ctx);
        }
    }

    return XCCL_OK;
}

void xccl_context_destroy(xccl_context_h context)
{
    xccl_tl_context_t *tl_ctx;
    int               i;

    for (i = 0; i < context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        tl_ctx->lib->team_context_destroy(tl_ctx);
    }

    free(context->tl_ctx);
    free(context);
}

xccl_status_t xccl_context_config_read(xccl_lib_h lib, const char *env_prefix,
                                       const char *filename,
                                       xccl_context_config_t **config_p)
{
    xccl_tl_context_t *tl_ctx;
    int               i;
    char full_prefix[128] = "XCCL_";
    ucs_status_t status;
    xccl_context_config_t *config;
    int                   env_prefix_len;

    config          = (xccl_context_config_t*)malloc(sizeof(xccl_context_config_t));
    if (config == NULL) {
        goto err_config;
    }

    config->configs = (xccl_tl_context_config_t**)malloc(lib->n_libs_opened * sizeof(xccl_tl_context_config_t*));
    if (config->configs == NULL) {
        goto err_configs;
    }

    for(i = 0; i < lib->n_libs_opened; i++) {
        if (lib->libs[i]->tl_context_config.table == NULL) {
            continue;
        }
        config->configs[i] = (xccl_tl_context_config_t*)malloc(lib->libs[i]->tl_context_config.size);
        if (config->configs[i] == NULL) {
            goto err_config_i;
        }

        config->configs[i]->env_prefix = NULL;
        if ((env_prefix != NULL) && (strlen(env_prefix) > 0)) {
            config->configs[i]->env_prefix = strdup(env_prefix);
            if (config->configs[i]->env_prefix == NULL) {
                goto err_prefix;
            }

            snprintf(full_prefix, sizeof(full_prefix), "%s_%s", env_prefix, "XCCL_");
        }
        status = ucs_config_parser_fill_opts(config->configs[i],
                                             lib->libs[i]->tl_context_config.table,
                                             full_prefix,
                                             lib->libs[i]->tl_context_config.prefix,
                                             0);    
    }
    config->n_tl_cfg = lib->n_libs_opened;
    config->lib      = lib;
    *config_p = config;

    return XCCL_OK;

err_prefix:
    free(config->configs[i]);

err_config_i:
    for(i = i - 1;i >= 0; i--) {
        free(config->configs[i]);
        if (config->configs[i]->env_prefix != NULL) {
            free(config->configs[i]->env_prefix);
        }
    }
err_configs:
    free(config->configs);

err_config:
    free(config);

    return XCCL_ERR_NO_MEMORY;
}

xccl_status_t xccl_context_config_modify(xccl_tl_id_t *tl_id,
                                         xccl_context_config_t *config,
                                         const char *name, const char *value)
{
    int i;

    for(i = 0; i < config->n_tl_cfg; i++)
    {
        if (config->lib->libs[i]->id == *tl_id) {
            return ucs_config_parser_set_value(config->configs[i],
                                               config->lib->libs[i]->tl_context_config.table,
                                               name, value);
        }
    }
    xccl_debug("No tl found to modify config");
    return XCCL_ERR_INVALID_PARAM;
}


void xccl_context_config_release(xccl_context_config_t *config)
{
    int i;

    for(i = 0; i < config->lib->n_libs_opened; i++) {
        ucs_config_parser_release_opts(config->configs[i],
                                       config->lib->libs[i]->tl_context_config.table);
        if (config->configs[i]->env_prefix != NULL) {
            free(config->configs[i]->env_prefix);
        }
        free(config->configs[i]);
    }

    free(config->configs);
    free(config);
}
