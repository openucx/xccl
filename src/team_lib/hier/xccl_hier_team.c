/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_hier_context.h"
#include "xccl_hier_team.h"
#include "core/xccl_team.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static int xccl_sbgp_rank_to_context(int rank, void *rank_mapper_ctx) {
    xccl_sbgp_t *sbgp = (xccl_sbgp_t*)rank_mapper_ctx;
    return xccl_sbgp_rank2ctx(sbgp, rank);
}

static int xccl_sbgp_rank_to_team(int rank, void *rank_mapper_ctx) {
    xccl_sbgp_t *sbgp = (xccl_sbgp_t*)rank_mapper_ctx;
    return xccl_sbgp_rank2team(sbgp, rank);
}

static int
oob_sbgp_allgather(void *sbuf, void *rbuf, size_t len,
                   int myrank, xccl_ep_range_t r, void *coll_context, void **req) {
    xccl_sbgp_t *sbgp = (xccl_sbgp_t*)coll_context;
    xccl_team_t *team = sbgp->team;
    assert(r.type == XCCL_EP_RANGE_UNDEFINED);
    xccl_ep_range_t range = {
        .type      = XCCL_EP_RANGE_CB,
        .ep_num    = sbgp->group_size,
        .cb.cb     = xccl_sbgp_rank_to_team,
        .cb.cb_ctx = (void*)sbgp,
    };
    team->params.oob.allgather(sbuf, rbuf, len, sbgp->group_rank,
                                     range, team->params.oob.coll_context,
                                     req);
    return 0;
}

static xccl_status_t xccl_hier_create_pair(xccl_sbgp_t *sbgp, xccl_hier_team_t *team,
                                           xccl_tl_id_t tl_id, xccl_hier_pair_type_t pair) {
    xccl_status_t status;
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx, xccl_hier_context_t);
    if (sbgp->status != XCCL_SBGP_ENABLED) {
        return 0;
    }

    xccl_oob_collectives_t oob = {
        .allgather    = oob_sbgp_allgather,
        .req_test     = team->super.params.oob.req_test,
        .req_free     = team->super.params.oob.req_free,
        .coll_context = (void*)sbgp,
        .rank         = sbgp->group_rank,
        .size         = sbgp->group_size,
    };

    xccl_team_params_t team_params = {
        .range.type      = XCCL_EP_RANGE_CB,
        .range.cb.cb     = xccl_sbgp_rank_to_context,
        .range.cb.cb_ctx = (void*)sbgp,
        .oob             = oob,
    };

    team->pairs[pair] = (xccl_hier_pair_t*)malloc(sizeof(xccl_hier_pair_t));
    status = xccl_team_create_post(ctx->tls[ucs_ilog2(tl_id)].xccl_ctx, &team_params,
                                   &team->pairs[pair]->team);
    if (status != XCCL_OK) {
        xccl_hier_warn("Failed to create team for TL %s", xccl_tl_str(tl_id));
        free(team->pairs[pair]);
        team->pairs[pair] = NULL;
        return status;
    }
    while (XCCL_INPROGRESS ==
           xccl_team_create_test(team->pairs[pair]->team)) {;}
    team->pairs[pair]->sbgp = sbgp;
    return XCCL_OK;
}

xccl_status_t xccl_hier_team_create_post(xccl_tl_context_t *context,
                                         xccl_team_params_t *params,
                                         xccl_team_t *base_team,
                                         xccl_tl_team_t **team)
{
    //TODO need to make this non blocking + team_hier_wait
    xccl_status_t       status = XCCL_OK;
    xccl_hier_context_t *ctx   = ucs_derived_of(context, xccl_hier_context_t);
    int                 size   = params->oob.size;
    int                 rank   = params->oob.rank;
    int                 i;
    xccl_hier_team_t    *hier_team;

    hier_team = (xccl_hier_team_t*)calloc(1, sizeof(xccl_hier_team_t));
    XCCL_TEAM_SUPER_INIT(hier_team->super, context, params, base_team);

    hier_team->no_socket = 0;
    for (i=0; i<size; i++) {
        if (base_team->topo->topo->procs[xccl_team_rank2ctx(base_team, i)].socketid < 0) {
            hier_team->no_socket = 1;
            break;
        }
    }
    if (!hier_team->no_socket) {
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_SOCKET), hier_team,
                              XCCL_TL_UCX, XCCL_HIER_PAIR_SOCKET_UCX);
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_SOCKET_LEADERS), hier_team,
                              XCCL_TL_UCX, XCCL_HIER_PAIR_SOCKET_LEADERS_UCX);
    } else {
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE), hier_team,
                              XCCL_TL_UCX, XCCL_HIER_PAIR_NODE_UCX);
    }
    xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE_LEADERS), hier_team,
                          XCCL_TL_UCX, XCCL_HIER_PAIR_NODE_LEADERS_UCX);

    if (ctx->tls[ucs_ilog2(XCCL_TL_SHMSEG)].enabled) {
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_SOCKET), hier_team,
                              XCCL_TL_SHMSEG, XCCL_HIER_PAIR_SOCKET_SHMSEG);
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_SOCKET_LEADERS), hier_team,
                              XCCL_TL_SHMSEG, XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG);
    }

    if (ctx->tls[ucs_ilog2(XCCL_TL_SHARP)].enabled) {
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE_LEADERS), hier_team,
                              XCCL_TL_SHARP, XCCL_HIER_PAIR_NODE_LEADERS_SHARP);
    }

    if (ctx->tls[ucs_ilog2(XCCL_TL_VMC)].enabled) {
        xccl_hier_create_pair(xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE_LEADERS), hier_team,
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
    xccl_hier_team_t *hier_team = ucs_derived_of(team, xccl_hier_team_t);
    int i;

    for (i=0; i<XCCL_HIER_PAIR_LAST; i++) {
        if (hier_team->pairs[i]) {
            xccl_team_destroy(hier_team->pairs[i]->team);
            free(hier_team->pairs[i]);
        }
    }

    free(hier_team);
    return XCCL_OK;
}
