/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "tccl_team_lib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
tccl_status_t tccl_lib_finalize(tccl_lib_h lib);
static tccl_status_t tccl_create_team_context(tccl_team_lib_h lib,
                                              tccl_context_config_h config,
                                              tccl_tl_context_t **team_ctx)
{
    return lib->create_team_context(lib, config, team_ctx);
}

static tccl_status_t find_tlib_by_name(tccl_lib_t *lib, const char *tlib_name,
                                       tccl_team_lib_t **tlib) {
    int i;
    for (i=0; i<lib->n_libs_opened; i++) {
        if (0 == strcmp(tlib_name, lib->libs[i]->name)) {
            *tlib = lib->libs[i];
            return TCCL_OK;
        }
    }
    *tlib = NULL;
    return TCCL_ERR_NO_ELEM;
}

tccl_status_t tccl_create_context(tccl_lib_t *lib, const tccl_config_t *config,
                                  tccl_context_t **context)
{
    tccl_context_t *ctx = malloc(sizeof(*ctx));
    char *default_tls[5] = {"ucx", "sharp", "vmc", "shmseg", NULL};
    tccl_team_lib_t *tlib;
    tccl_tl_context_t *tl_ctx;
    char **tls;
    int num_tls = 0;
    ctx->lib = lib;
    memcpy(&ctx->cfg, &config->ctx_config, sizeof(ctx->cfg));
    if (config->tls[0]) {
        tls = config->tls;
    } else {
        tls = default_tls;
    }
    assert(tls[0]);
    while (tls[num_tls]) {
        num_tls++;
    }
    ctx->tl_ctx = (tccl_tl_context_t**)malloc(sizeof(tccl_tl_context_t*)*num_tls);
    ctx->n_tl_ctx = 0;
    num_tls = 0;
    while (tls[num_tls++]) {
        if (TCCL_OK != find_tlib_by_name(lib, tls[num_tls-1], &tlib)) {
            continue;;
        }
        if (TCCL_OK != tccl_create_team_context(tlib, &ctx->cfg, &tl_ctx)) {
            continue;
        } else {
            /* fprintf(stderr, "Created ctx %s, prio %d\n", tl_ctx->lib->name, tl_ctx->lib->priority); */
            ctx->tl_ctx[ctx->n_tl_ctx++] = tl_ctx;
        }
    }
    *context = ctx;
    return TCCL_OK;
}

tccl_status_t tccl_cleanup(tccl_context_t *context)
{
    int i;
    tccl_tl_context_t *tl_ctx;
    for (i=0; i<context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        tl_ctx->lib->destroy_team_context(tl_ctx);
    }
    tccl_lib_finalize(context->lib);
    free(context->tl_ctx);
    free(context);
    return TCCL_OK;
}
