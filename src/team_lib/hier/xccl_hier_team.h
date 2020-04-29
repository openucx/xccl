/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_HIER_TEAM_H_
#define XCCL_HIER_TEAM_H_
#include "xccl_hier_lib.h"
#include "xccl_hier_sbgp.h"
#include "xccl_hier_context.h"

typedef struct xccl_hier_pair {
    xccl_team_h team;
    sbgp_t     *sbgp;
} xccl_hier_pair_t;

typedef enum {
    XCCL_HIER_PAIR_NODE_UCX,
    XCCL_HIER_PAIR_SOCKET_UCX,
    XCCL_HIER_PAIR_NODE_LEADERS_UCX,
    XCCL_HIER_PAIR_SOCKET_LEADERS_UCX,
    XCCL_HIER_PAIR_NODE_SHMSEG,
    XCCL_HIER_PAIR_SOCKET_SHMSEG,
    XCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG,
    XCCL_HIER_PAIR_NODE_LEADERS_SHARP,
    XCCL_HIER_PAIR_NODE_LEADERS_VMC,
    XCCL_HIER_PAIR_LAST,
} xccl_hier_pair_type_t;

typedef struct xccl_hier_team {
    xccl_tl_team_t             super;
    sbgp_t                     sbgps[SBGP_LAST];
    xccl_hier_pair_t           *pairs[XCCL_HIER_PAIR_LAST];
    int                        node_leader_rank;
} xccl_hier_team_t;

xccl_status_t xccl_hier_team_create_post(xccl_tl_context_t *context,
                                         xccl_team_params_t *config,
                                         xccl_tl_team_t **team);
xccl_status_t xccl_hier_team_create_test(xccl_tl_team_t *team);
xccl_status_t xccl_hier_team_destroy(xccl_tl_team_t *team);


static inline int xccl_hier_team_rank2ctx(xccl_hier_team_t *team, int rank)
{
    return xccl_range_to_rank(team->super.params.range, rank);
}

static inline int sbgp_rank2ctx(sbgp_t *sbgp, int rank)
{
    return xccl_range_to_rank(sbgp->hier_team->super.params.range,
                              sbgp_rank2team(sbgp, rank));
}

static inline int is_rank_on_local_node(int rank, xccl_hier_team_t *team)
{
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx, xccl_hier_context_t);
    return ctx->procs[xccl_hier_team_rank2ctx(team, rank)].node_hash
        == ctx->local_proc.node_hash;
}

static inline int is_rank_on_local_socket(int rank, xccl_hier_team_t *team)
{
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx, xccl_hier_context_t);
    if (ctx->local_proc.socketid < 0) {
        return 0;
    }
    xccl_hier_proc_data_t *proc = &ctx->procs[xccl_hier_team_rank2ctx(team, rank)];
    return proc->node_hash == ctx->local_proc.node_hash &&
        proc->socketid == ctx->local_proc.socketid;
}

#endif
