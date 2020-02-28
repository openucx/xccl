/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "xccl_team_lib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
xccl_status_t xccl_lib_finalize(xccl_lib_h lib);
static xccl_status_t xccl_create_team_context(xccl_team_lib_h lib,
                                              xccl_context_config_h config,
                                              xccl_tl_context_t **team_ctx)
{
    return lib->create_team_context(lib, config, team_ctx);
}

static xccl_status_t find_tlib_by_name(xccl_lib_t *lib, const char *tlib_name,
                                       xccl_team_lib_t **tlib) {
    int i;
    for (i=0; i<lib->n_libs_opened; i++) {
        if (0 == strcmp(tlib_name, lib->libs[i]->name)) {
            *tlib = lib->libs[i];
            return XCCL_OK;
        }
    }
    *tlib = NULL;
    return XCCL_ERR_NO_ELEM;
}

xccl_status_t xccl_create_context(xccl_lib_t *lib, const xccl_config_t *config,
                                  xccl_context_t **context)
{
    xccl_context_t *ctx = malloc(sizeof(*ctx));
    char *default_tls = "ucx,sharp,vmc,shmseg";
    xccl_team_lib_t *tlib;
    xccl_tl_context_t *tl_ctx;
    char *tls, *tl, *saveptr;
    int i;
    int num_tls = 1;
    ctx->lib = lib;
    memcpy(&ctx->cfg, &config->ctx_config, sizeof(ctx->cfg));
    if (config->tls) {
        tls = strdup(config->tls);
    } else {
        tls = strdup(default_tls);
    }
    for (i=0; i<strlen(tls); i++) {
        if (tls[i] == ',') {
            num_tls++;
        }
    }
    ctx->tl_ctx = (xccl_tl_context_t**)malloc(sizeof(xccl_tl_context_t*)*num_tls);
    ctx->n_tl_ctx = 0;
    for (tl = strtok_r(tls, ",", &saveptr); tl != NULL;
         tl = strtok_r(NULL, ",", &saveptr)) {
        if (XCCL_OK == find_tlib_by_name(lib, tl, &tlib)) {
            if (XCCL_OK == xccl_create_team_context(tlib, &ctx->cfg, &tl_ctx)) {
                /* fprintf(stderr, "Created ctx %s, prio %d\n", tl_ctx->lib->name, tl_ctx->lib->priority); */
                ctx->tl_ctx[ctx->n_tl_ctx++] = tl_ctx;
            }
        }
        tl = strtok(NULL, ",");
    }
    free(tls);

    *context = ctx;
    return XCCL_OK;
}

xccl_status_t xccl_cleanup(xccl_context_t *context)
{
    int i;
    xccl_tl_context_t *tl_ctx;
    for (i=0; i<context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        tl_ctx->lib->destroy_team_context(tl_ctx);
    }
    xccl_lib_finalize(context->lib);
    free(context->tl_ctx);
    free(context);
    return XCCL_OK;
}
