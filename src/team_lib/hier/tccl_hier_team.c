/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "tccl_hier_context.h"
#include "tccl_hier_team.h"
#include "tccl_hier_sbgp.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static int tccl_sbgp_rank_to_context(int rank, void *rank_mapper_ctx) {
    sbgp_t *sbgp = (sbgp_t*)rank_mapper_ctx;
    return sbgp_rank2ctx(sbgp, rank);
}

static int tccl_sbgp_rank_to_team(int rank, void *rank_mapper_ctx) {
    sbgp_t *sbgp = (sbgp_t*)rank_mapper_ctx;
    return sbgp_rank2team(sbgp, rank);
}

static int
oob_sbgp_allgather(void *sbuf, void *rbuf, size_t len,
                   int myrank, tccl_ep_range_t r, void *coll_context) {
    sbgp_t *sbgp = (sbgp_t*)coll_context;
    tccl_hier_team_t *team = sbgp->hier_team;
    assert(r.type == TCCL_EP_RANGE_UNDEFINED);
    tccl_ep_range_t range = {
        .type      = TCCL_EP_RANGE_CB,
        .ep_num    = sbgp->group_size,
        .cb.cb     = tccl_sbgp_rank_to_team,
        .cb.cb_ctx = (void*)sbgp,
    };
    team->super.oob.allgather(sbuf, rbuf, len, team->super.oob.rank,
                              range, team->super.oob.coll_context);
    return 0;
}

static tccl_status_t tccl_hier_create_pair(sbgp_t *sbgp, tccl_hier_team_t *team,
                                           tccl_tl_id_t tl_id, tccl_hier_pair_type_t pair) {
    tccl_hier_context_t *ctx = tccl_derived_of(team->super.ctx, tccl_hier_context_t);
    if (sbgp->status != SBGP_ENABLED) {
        return 0;
    }

    tccl_team_config_t team_config = {
        .range.type      = TCCL_EP_RANGE_CB,
        .range.cb.cb     = tccl_sbgp_rank_to_context,
        .range.cb.cb_ctx = (void*)sbgp,
    };

    tccl_oob_collectives_t oob = {
        .allgather  = oob_sbgp_allgather,
        .coll_context = (void*)sbgp,
        .rank = sbgp->group_rank,
        .size = sbgp->group_size,
    };

    team->pairs[pair] = (tccl_hier_pair_t*)malloc(sizeof(tccl_hier_pair_t));
    tccl_team_create_post(ctx->tls[tl_id].tccl_ctx, &team_config,
                         oob, &team->pairs[pair]->team);
    team->pairs[pair]->sbgp = sbgp;
    return TCCL_OK;
}

tccl_status_t tccl_hier_team_create_post(tccl_tl_context_t *context, tccl_team_config_t *config,
                                         tccl_oob_collectives_t oob, tccl_tl_team_t **team)
{
    //TODO need to make this non blocking + team_hier_wait
    tccl_status_t status      = TCCL_OK;
    tccl_hier_context_t *ctx = tccl_derived_of(context, tccl_hier_context_t);
    int i, size = oob.size, rank = oob.rank;
    tccl_hier_team_t *hier_team;

    hier_team = (tccl_hier_team_t*)calloc(1, sizeof(tccl_hier_team_t));
    TCCL_TEAM_SUPER_INIT(hier_team->super, context, config, oob);

    for (i=0; i<SBGP_LAST; i++) {
        hier_team->sbgps[i].status = SBGP_DISABLED;
    }

    sbgp_create(hier_team, SBGP_NODE);
    sbgp_create(hier_team, SBGP_SOCKET);
    sbgp_create(hier_team, SBGP_NODE_LEADERS);
    sbgp_create(hier_team, SBGP_SOCKET_LEADERS);

    tccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET], hier_team,
                          TCCL_TL_UCX, TCCL_HIER_PAIR_SOCKET_UCX);
    tccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET_LEADERS], hier_team,
                          TCCL_TL_UCX, TCCL_HIER_PAIR_SOCKET_LEADERS_UCX);
    tccl_hier_create_pair(&hier_team->sbgps[SBGP_NODE_LEADERS], hier_team,
                          TCCL_TL_UCX, TCCL_HIER_PAIR_NODE_LEADERS_UCX);

    if (ctx->tls[TCCL_TL_SHMSEG].enabled) {
        tccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET], hier_team,
                              TCCL_TL_SHMSEG, TCCL_HIER_PAIR_SOCKET_SHMSEG);
        tccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET_LEADERS], hier_team,
                              TCCL_TL_SHMSEG, TCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG);
    }

    if (ctx->tls[TCCL_TL_SHARP].enabled) {
        tccl_hier_create_pair(&hier_team->sbgps[SBGP_NODE_LEADERS], hier_team,
                              TCCL_TL_SHARP, TCCL_HIER_PAIR_NODE_LEADERS_SHARP);
    }

    if (ctx->tls[TCCL_TL_VMC].enabled) {
        tccl_hier_create_pair(&hier_team->sbgps[SBGP_NODE_LEADERS], hier_team,
                              TCCL_TL_VMC, TCCL_HIER_PAIR_NODE_LEADERS_VMC);
    }

    *team = &hier_team->super;
    return TCCL_OK;
}

tccl_status_t tccl_hier_team_destroy(tccl_tl_team_t *team)
{
    tccl_hier_team_t *hier_team = tccl_derived_of(team, tccl_hier_team_t);
    int i;

    for (i=0; i<TCCL_HIER_PAIR_LAST; i++) {
        if (hier_team->pairs[i]) {
            tccl_team_destroy(hier_team->pairs[i]->team);
            free(hier_team->pairs[i]);
        }
    }

    for (i=0; i<SBGP_LAST; i++) {
        if (SBGP_ENABLED == hier_team->sbgps[i].status) {
            sbgp_cleanup(&hier_team->sbgps[i]);
        }
    }
    free(hier_team);
    return TCCL_OK;
}
