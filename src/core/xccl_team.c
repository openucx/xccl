/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"

#include <xccl_team.h>
#include <ucs/memory/memory_type.h>
#include <stdlib.h>
#include <stdio.h>

static int compare_teams_by_priority(const void* t1, const void* t2)
{
    const xccl_tl_team_t** team1 = (const xccl_tl_team_t**)t1;
    const xccl_tl_team_t** team2 = (const xccl_tl_team_t**)t2;
    return (*team2)->ctx->lib->priority - (*team1)->ctx->lib->priority;
}


xccl_status_t xccl_team_create_post(xccl_context_h context,
                                    xccl_team_params_t *params,
                                    xccl_team_t **xccl_team)
{
    int i;
    int n_ctx = context->n_tl_ctx;
    xccl_collective_type_t c;
    xccl_team_t *team;
    xccl_tl_context_t *tl_ctx;
    xccl_status_t status;

    *xccl_team = NULL;
    if (context->n_tl_ctx < 1) {
        xccl_error("No library contexts available");
        return XCCL_ERR_NO_MESSAGE;
    }

    team = (xccl_team_t*)malloc(sizeof(*team) +
                                sizeof(xccl_tl_team_t*)*(n_ctx-1));
    team->ctx = context;
    team->n_teams = 0;
    memcpy(&team->params, params, sizeof(xccl_team_params_t));
    for (i=0; i<context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        status = tl_ctx->lib->team_create_post(tl_ctx, params,
                                               &team->tl_teams[team->n_teams]);
        if (status != XCCL_OK) {
            continue;
        }
        status = tl_ctx->lib->team_create_test(team->tl_teams[team->n_teams]);
        team->n_teams++;
        if (status == XCCL_INPROGRESS) {
            /* workaround to fix oob allgather issue if multiple teams use it
               simultaneously*/
            break;
        }
    }
    if (team->n_teams == 0) {
        xccl_warn("no teams were opened");
        return XCCL_ERR_NO_MESSAGE;
    }
    team->last_team_create_posted = i;
    team->status                  = XCCL_INPROGRESS;
    *xccl_team                    = team;
    return XCCL_OK;
}

xccl_status_t xccl_team_create_test(xccl_team_t *team)
{
    int i, c, m;
    xccl_tl_context_t *tl_ctx;
    xccl_status_t status;

    tl_ctx = team->ctx->tl_ctx[team->n_teams - 1];
    status = tl_ctx->lib->team_create_test(team->tl_teams[team->n_teams-1]);
    if (status != XCCL_OK) {
        return status;
    }

    for (i = team->last_team_create_posted + 1; i < team->ctx->n_tl_ctx; i++) {
        tl_ctx = team->ctx->tl_ctx[i];
        status = tl_ctx->lib->team_create_post(tl_ctx, &team->params,
                                               &team->tl_teams[team->n_teams]);
        team->last_team_create_posted = i;
        if (status != XCCL_OK) {
            continue;
        }
        status = tl_ctx->lib->team_create_test(team->tl_teams[team->n_teams]);
        team->n_teams++;
        if (status == XCCL_INPROGRESS) {
            /* workaround to fix oob allgather issue if multiple teams use it
               simultaneously*/
            return XCCL_INPROGRESS;
        }
    }
    qsort(team->tl_teams, team->n_teams, sizeof(xccl_tl_team_t*),
          compare_teams_by_priority);
    for (m = 0; m < UCS_MEMORY_TYPE_LAST; m++) {
        for (c = 0; c < XCCL_COLL_LAST; c++) {
            team->coll_team_id[c][m] = -1;
            for (i=0; i<team->n_teams; i++) {
                if ((team->tl_teams[i]->ctx->lib->params.coll_types & UCS_BIT(c)) &&
                    (team->tl_teams[i]->ctx->lib->mem_types & UCS_BIT(m))) {
                    team->coll_team_id[c][m] = i;
                    break;
                }
            }
        }
    }
    team->status = XCCL_OK;
    /* TODO: check if some teams are never used after selection and clean them up */
    return XCCL_OK;
}

void xccl_team_destroy(xccl_team_t *team)
{
    xccl_tl_context_t *tl_ctx;
    int               i;

    if (team->status != XCCL_OK) {
        xccl_error("team %p is used before team_create is completed", team);
        return;
    }

    for (i=0; i<team->n_teams; i++) {
        tl_ctx = team->tl_teams[i]->ctx;
        tl_ctx->lib->team_destroy(team->tl_teams[i]);
    }
    free(team);
}
