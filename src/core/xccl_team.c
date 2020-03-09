/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "xccl_team_lib.h"
#include <stdlib.h>
#include <stdio.h>

ucs_config_field_t xccl_team_lib_config_table[] = {
  {"LOG_LEVEL", "warn",
  "XCCL logging level. Messages with a level higher or equal to the selected "
  "will be printed.\n"
  "Possible values are: fatal, error, warn, info, debug, trace, data, func, poll.",
  ucs_offsetof(xccl_lib_config_t, log_component),
  UCS_CONFIG_TYPE_COMP},

  {NULL}
};

static int compare_teams_by_priority(const void* t1, const void* t2)
{
    const xccl_tl_team_t** team1 = (const xccl_tl_team_t**)t1;
    const xccl_tl_team_t** team2 = (const xccl_tl_team_t**)t2;
    return (*team2)->ctx->lib->priority - (*team1)->ctx->lib->priority;
}


xccl_status_t xccl_team_create_post(xccl_context_t *context,
                                    xccl_team_config_t *config,
                                    xccl_oob_collectives_t oob, xccl_team_t **xccl_team)
{
    int i;
    int n_ctx = context->n_tl_ctx;
    xccl_collective_type_t c;
    xccl_team_t *team;
    xccl_tl_context_t *tl_ctx;
    xccl_status_t status;

    *xccl_team = NULL;
    if (context->n_tl_ctx < 1) {
        return XCCL_ERR_NO_MESSAGE;
    }

    team = (xccl_team_t*)malloc(sizeof(*team) +
                                sizeof(xccl_tl_team_t*)*(n_ctx-1));
    team->ctx = context;
    team->n_teams = 0;

    for (i=0; i<context->n_tl_ctx; i++) {
        tl_ctx = context->tl_ctx[i];
        status = tl_ctx->lib->team_create_post(tl_ctx, config,
                                               oob, &team->tl_teams[team->n_teams]);
        if (XCCL_OK == status) {
            /* fprintf(stderr, "Created team %s\n", team->tl_teams[team->n_teams]->ctx->lib->name); */
            team->n_teams++;
        }
    }
    if (team->n_teams == 0) {
        free(team);
        return XCCL_ERR_NO_MESSAGE;
    }
    qsort(team->tl_teams, team->n_teams, sizeof(xccl_tl_team_t*),
          compare_teams_by_priority);
    for (c = 0; c < XCCL_COLL_LAST; c++) {
        for (i=0; i<team->n_teams; i++) {
            if (team->tl_teams[i]->ctx->lib->params.coll_types &
                XCCL_BIT(c)) {
                team->coll_team_id[c] = i;
                break;
            }
        }
    }
    /* TODO: check if some teams are never used after selection and clean them up */
    *xccl_team = team;
    return XCCL_OK;
}

xccl_status_t xccl_team_destroy(xccl_team_t *team)
{
    int i;
    xccl_tl_context_t *tl_ctx;
    for (i=0; i<team->n_teams; i++) {
        tl_ctx = team->tl_teams[i]->ctx;
        tl_ctx->lib->team_destroy(team->tl_teams[i]);
    }
    free(team);
    return XCCL_OK;
}
