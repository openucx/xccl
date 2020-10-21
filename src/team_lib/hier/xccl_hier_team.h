/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_HIER_TEAM_H_
#define XCCL_HIER_TEAM_H_
#include "xccl_hier_lib.h"
#include "xccl_hier_context.h"
#include "topo/xccl_topo.h"
#include "core/xccl_team.h"

typedef struct xccl_hier_pair {
    xccl_team_h team;
    xccl_sbgp_t *sbgp;
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
    xccl_hier_pair_t           *pairs[XCCL_HIER_PAIR_LAST];
    int                        node_leader_rank;
    int                        no_socket;
} xccl_hier_team_t;

xccl_status_t xccl_hier_team_create_post(xccl_tl_context_t *context,
                                         xccl_team_params_t *config,
                                         xccl_team_t *base_team,
                                         xccl_tl_team_t **team);
xccl_status_t xccl_hier_team_create_test(xccl_tl_team_t *team);
xccl_status_t xccl_hier_team_destroy(xccl_tl_team_t *team);


static inline int xccl_hier_team_rank2ctx(xccl_hier_team_t *team, int rank)
{
    return xccl_range_to_rank(team->super.params.range, rank);
}

static inline int is_rank_on_local_node(int rank, xccl_hier_team_t *team)
{
    return xccl_rank_on_local_node(rank, team->super.base_team);
}

static inline int is_rank_on_local_socket(int rank, xccl_hier_team_t *team)
{
    return xccl_rank_on_local_socket(rank, team->super.base_team);
}

#endif
