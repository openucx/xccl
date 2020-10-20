/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_TEAM_H_
#define XCCL_TEAM_H_

#include <api/xccl.h>
#include <xccl_context.h>
#include <xccl_team_lib.h>
#include <ucs/memory/memory_type.h>
#include "topo/xccl_topo.h"

#define XCCL_CHECK_TEAM(_team)                                                    \
    do {                                                                          \
        if (_team->status != XCCL_OK) {                                           \
            xccl_error("team %p is used before team_create is completed", _team); \
            return XCCL_ERR_INVALID_PARAM;                                        \
        }                                                                         \
    } while(0)

typedef struct xccl_team {
    xccl_context_t *ctx;
    int            coll_team_id[XCCL_COLL_LAST][UCS_MEMORY_TYPE_LAST];
    int            n_teams;
    xccl_status_t  status;
    xccl_team_params_t params;
    xccl_team_topo_t *topo;
    xccl_tl_team_t *tl_teams[1];
} xccl_team_t;

static inline int xccl_team_rank2ctx(xccl_team_t *team, int rank)
{
    return xccl_range_to_rank(team->params.range, rank);
}

static inline int xccl_rank_on_local_node(int rank, xccl_team_t *team)
{
    return team->topo->topo->procs[xccl_team_rank2ctx(team, rank)].node_hash
        == team->topo->topo->local_proc.node_hash;
}

static inline int xccl_rank_on_local_socket(int rank, xccl_team_t *team)
{
    if (team->topo->topo->local_proc.socketid < 0) {
        return 0;
    }
    xccl_proc_data_t *proc = &team->topo->topo->procs[xccl_team_rank2ctx(team, rank)];
    return proc->node_hash == team->topo->topo->local_proc.node_hash &&
        proc->socketid == team->topo->topo->local_proc.socketid;
}

#endif
