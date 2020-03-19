/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_hier_context.h"
#include "xccl_hier_team.h"
#include "xccl_hier_sbgp.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static int xccl_sbgp_rank_to_context(int rank, void *rank_mapper_ctx) {
    sbgp_t *sbgp = (sbgp_t*)rank_mapper_ctx;
    return sbgp_rank2ctx(sbgp, rank);
}

static int xccl_sbgp_rank_to_team(int rank, void *rank_mapper_ctx) {
    sbgp_t *sbgp = (sbgp_t*)rank_mapper_ctx;
    return sbgp_rank2team(sbgp, rank);
}

static int
oob_sbgp_allgather(void *sbuf, void *rbuf, size_t len,
                   int myrank, xccl_ep_range_t r, void *coll_context, void **req) {
    sbgp_t *sbgp = (sbgp_t*)coll_context;
    xccl_hier_team_t *team = sbgp->hier_team;
    assert(r.type == XCCL_EP_RANGE_UNDEFINED);
    xccl_ep_range_t range = {
        .type      = XCCL_EP_RANGE_CB,
        .ep_num    = sbgp->group_size,
        .cb.cb     = xccl_sbgp_rank_to_team,
        .cb.cb_ctx = (void*)sbgp,
    };
    team->super.oob.allgather(sbuf, rbuf, len, sbgp->group_rank,
                              range, team->super.oob.coll_context, req);
    return 0;
}

static xccl_status_t xccl_hier_create_pair(sbgp_t *sbgp, xccl_hier_team_t *team,
                                           xccl_tl_id_t tl_id, xccl_hier_pair_type_t pair) {
    xccl_hier_context_t *ctx = xccl_derived_of(team->super.ctx, xccl_hier_context_t);
    if (sbgp->status != SBGP_ENABLED) {
        return 0;
    }

    xccl_team_config_t team_config = {
        .range.type      = XCCL_EP_RANGE_CB,
        .range.cb.cb     = xccl_sbgp_rank_to_context,
        .range.cb.cb_ctx = (void*)sbgp,
    };

    xccl_oob_collectives_t oob = {
        .allgather  = oob_sbgp_allgather,
        .coll_context = (void*)sbgp,
        .rank = sbgp->group_rank,
        .size = sbgp->group_size,
    };

    team->pairs[pair] = (xccl_hier_pair_t*)malloc(sizeof(xccl_hier_pair_t));
    xccl_team_create_post(ctx->tls[tl_id].xccl_ctx, &team_config,
                         oob, &team->pairs[pair]->team);
    team->pairs[pair]->sbgp = sbgp;
    return XCCL_OK;
}

xccl_status_t xccl_hier_team_create_post(xccl_tl_context_t *context, xccl_team_config_t *config,
                                         xccl_oob_collectives_t oob, xccl_tl_team_t **team)
{
    //TODO need to make this non blocking + team_hier_wait
    xccl_status_t status      = XCCL_OK;
    xccl_hier_context_t *ctx = xccl_derived_of(context, xccl_hier_context_t);
    int i, size = oob.size, rank = oob.rank;
    xccl_hier_team_t *hier_team;

    hier_team = (xccl_hier_team_t*)calloc(1, sizeof(xccl_hier_team_t));
    XCCL_TEAM_SUPER_INIT(hier_team->super, context, config, oob);

    for (i=0; i<SBGP_LAST; i++) {
        hier_team->sbgps[i].status = SBGP_DISABLED;
    }

    sbgp_create(hier_team, SBGP_NODE);
    sbgp_create(hier_team, SBGP_SOCKET);
    sbgp_create(hier_team, SBGP_NODE_LEADERS);
    sbgp_create(hier_team, SBGP_SOCKET_LEADERS);

    xccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET], hier_team,
                          XCCL_TL_UCX, XCCL_HIER_PAIR_SOCKET_UCX);
    xccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET_LEADERS], hier_team,
                          XCCL_TL_UCX, XCCL_HIER_PAIR_SOCKET_LEADERS_UCX);
    xccl_hier_create_pair(&hier_team->sbgps[SBGP_NODE_LEADERS], hier_team,
                          XCCL_TL_UCX, XCCL_HIER_PAIR_NODE_LEADERS_UCX);

    if (ctx->tls[XCCL_TL_SHMSEG].enabled) {
        xccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET], hier_team,
                              XCCL_TL_SHMSEG, XCCL_HIER_PAIR_SOCKET_SHMSEG);
        xccl_hier_create_pair(&hier_team->sbgps[SBGP_SOCKET_LEADERS], hier_team,
                              XCCL_TL_SHMSEG, XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG);
    }

    if (ctx->tls[XCCL_TL_SHARP].enabled) {
        xccl_hier_create_pair(&hier_team->sbgps[SBGP_NODE_LEADERS], hier_team,
                              XCCL_TL_SHARP, XCCL_HIER_PAIR_NODE_LEADERS_SHARP);
    }

    if (ctx->tls[XCCL_TL_VMC].enabled) {
        xccl_hier_create_pair(&hier_team->sbgps[SBGP_NODE_LEADERS], hier_team,
                              XCCL_TL_VMC, XCCL_HIER_PAIR_NODE_LEADERS_VMC);
    }

    *team = &hier_team->super;
    return XCCL_OK;
}

xccl_status_t xccl_hier_team_create_test(xccl_tl_team_t *team)
{
    /*TODO implement true non-blocking */
    return XCCL_OK;
}

xccl_status_t xccl_hier_team_destroy(xccl_tl_team_t *team)
{
    xccl_hier_team_t *hier_team = xccl_derived_of(team, xccl_hier_team_t);
    int i;

    for (i=0; i<XCCL_HIER_PAIR_LAST; i++) {
        if (hier_team->pairs[i]) {
            xccl_team_destroy(hier_team->pairs[i]->team);
            free(hier_team->pairs[i]);
        }
    }

    for (i=0; i<SBGP_LAST; i++) {
        if (SBGP_ENABLED == hier_team->sbgps[i].status) {
            sbgp_cleanup(&hier_team->sbgps[i]);
        }
    }
    free(hier_team);
    return XCCL_OK;
}
