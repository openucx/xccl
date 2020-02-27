/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "tccl_team_lib.h"
#include <stdlib.h>

tccl_status_t tccl_team_create_post(tccl_context_t *context,
                                    tccl_team_config_t *config,
                                    tccl_oob_collectives_t oob, tccl_team_t **tccl_team)
{
    int i;
    int n_ctx = context->n_tl_ctx;
    tccl_team_t *team;
    tccl_tl_context_t *tl_ctx;
    tccl_status_t status;

    *tccl_team = NULL;
    if (context->n_tl_ctx < 1) {
        return TCCL_ERR_NO_MESSAGE;
    }

    team = (tccl_team_t*)malloc(sizeof(*team) +
                                sizeof(tccl_tl_team_t*)*(n_ctx-1));
    team->ctx = context;
    team->n_teams = 0;

    for (i=0; i<context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        status = tl_ctx->lib->team_create_post(tl_ctx, config,
                                               oob, &team->tl_teams[team->n_teams]);
        if (TCCL_OK == status) {
            team->n_teams++;
        }
    }
    if (team->n_teams == 0) {
        free(team);
        return TCCL_ERR_NO_MESSAGE;
    }
    *tccl_team = team;
    return TCCL_OK;
}

tccl_status_t tccl_team_destroy(tccl_team_t *team)
{
    int i;
    tccl_tl_context_t *tl_ctx;
    for (i=0; i<team->n_teams; i++) {
        tl_ctx = team->tl_teams[i]->ctx;
        tl_ctx->lib->team_destroy(team->tl_teams[i]);
    }
    free(team);
    return TCCL_OK;
}
