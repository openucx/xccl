/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_HIER_TEAM_H_
#define TCCL_HIER_TEAM_H_
#include "tccl_hier_lib.h"
#include "tccl_hier_sbgp.h"
#include "tccl_hier_context.h"

typedef struct tccl_hier_pair {
    tccl_team_h team;
    sbgp_t     *sbgp;
} tccl_hier_pair_t;

typedef enum {
    TCCL_HIER_PAIR_NODE_UCX,
    TCCL_HIER_PAIR_SOCKET_UCX,
    TCCL_HIER_PAIR_NODE_LEADERS_UCX,
    TCCL_HIER_PAIR_SOCKET_LEADERS_UCX,
    TCCL_HIER_PAIR_NODE_SHMSEG,
    TCCL_HIER_PAIR_SOCKET_SHMSEG,
    TCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG,
    TCCL_HIER_PAIR_NODE_LEADERS_SHARP,
    TCCL_HIER_PAIR_NODE_LEADERS_VMC,
    TCCL_HIER_PAIR_LAST,
} tccl_hier_pair_type_t;

typedef struct tccl_hier_team {
    tccl_tl_team_t     super;
    sbgp_t             sbgps[SBGP_LAST];
    tccl_hier_pair_t  *pairs[TCCL_HIER_PAIR_LAST];
} tccl_hier_team_t;

tccl_status_t tccl_hier_team_create_post(tccl_tl_context_t *context, tccl_team_config_t *config,
                                         tccl_oob_collectives_t oob, tccl_tl_team_t **team);
tccl_status_t tccl_hier_team_destroy(tccl_tl_team_t *team);

static inline int tccl_hier_team_rank2ctx(tccl_hier_team_t *team, int rank)
{
    return tccl_range_to_rank(team->super.cfg.range, rank);
}

static inline int sbgp_rank2ctx(sbgp_t *sbgp, int rank)
{
    return tccl_range_to_rank(sbgp->hier_team->super.cfg.range,
                              sbgp_rank2team(sbgp, rank));
}

static inline int is_rank_on_local_node(int rank, tccl_hier_team_t *team)
{
    tccl_hier_context_t *ctx = tccl_derived_of(team->super.ctx, tccl_hier_context_t);
    return ctx->procs[tccl_hier_team_rank2ctx(team, rank)].node_hash
        == ctx->local_proc.node_hash;
}

static inline int is_rank_on_local_socket(int rank, tccl_hier_team_t *team)
{
    tccl_hier_context_t *ctx = tccl_derived_of(team->super.ctx, tccl_hier_context_t);
    if (ctx->local_proc.socketid < 0) {
        return 0;
    }
    tccl_hier_proc_data_t *proc = &ctx->procs[tccl_hier_team_rank2ctx(team, rank)];
    return proc->node_hash == ctx->local_proc.node_hash &&
        proc->socketid == ctx->local_proc.socketid;
}

#endif
