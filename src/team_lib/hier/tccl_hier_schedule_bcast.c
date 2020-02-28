/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "tccl_hier_schedule.h"
#include "tccl_hier_team.h"
static inline int
root_at_socket(int root, sbgp_t *sbgp) {
    int i;
    int socket_root_rank = -1;
    for (i=0; i<sbgp->group_size; i++) {
        if (root == sbgp->rank_map[i]) {
            socket_root_rank = i;
            break;
        }
    }
    return socket_root_rank;
}

tccl_status_t build_bcast_schedule_3lvl(tccl_hier_team_t *team, coll_schedule_t **sched,
                                        tccl_coll_op_args_t coll, int node_leaders_pair)
{
    int have_node_leaders_group = (team->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group = (team->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (team->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);

    int node_leaders_group_exists = (team->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists = (team->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (team->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    sbgp_type_t top_sbgp = node_leaders_group_exists ? SBGP_NODE_LEADERS :
        (socket_leaders_group_exists ? SBGP_SOCKET_LEADERS : SBGP_SOCKET);

    int root = coll.root;
    int rank = team->super.oob.rank;
    int wroot = tccl_hier_team_rank2ctx(team, root);
    int root_on_local_node = is_rank_on_local_node(root, team);
    int root_on_local_socket = root_on_local_node &&
        is_rank_on_local_socket(root, team);
    tccl_hier_context_t *ctx = tccl_derived_of(team->super.ctx, tccl_hier_context_t);
    coll_schedule_single_dep_t *schedule = (coll_schedule_single_dep_t *)malloc(sizeof(*schedule));
    schedule->super.hier_team = team;
    schedule->super.type = TCCL_COLL_SCHED_SINGLE_DEP;
    int c = 0;

    int sock_leaders_pair = TCCL_HIER_PAIR_SOCKET_LEADERS_UCX;
    int sock_pair = TCCL_HIER_PAIR_SOCKET_UCX;

    if (coll.buffer_info.len <= 2048 && ctx->tls[TCCL_TL_SHMSEG].enabled) {
        sock_leaders_pair = TCCL_HIER_PAIR_SOCKET_LEADERS_SHMSEG;
        sock_pair = TCCL_HIER_PAIR_SOCKET_SHMSEG;
    }
    coll.alg.set_by_user = 0;
    schedule->dep_id = -1;
    if (rank == root) {
        schedule->dep_id = 0;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.root = ctx->procs[wroot].node_id;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[node_leaders_pair];
        if (coll.root != team->sbgps[SBGP_NODE_LEADERS].group_rank) {
            schedule->dep_id = c;
        }
        c++;
    }
    if (have_socket_leaders_group) {
        if (root_on_local_node) {
            coll.root = ctx->procs[wroot].socketid;
        } else {
            coll.root = 0;
        }
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[sock_leaders_pair];
        if (coll.root != team->sbgps[SBGP_SOCKET_LEADERS].group_rank) {
            schedule->dep_id = c;
        }
        c++;
    }
    if (have_socket_group) {
        if (root_on_local_socket) {
            coll.root = root_at_socket(root, &team->sbgps[SBGP_SOCKET]);
        } else {
            coll.root = 0;
        }
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[sock_pair];
        if (coll.root != team->sbgps[SBGP_SOCKET].group_rank) {
            schedule->dep_id = c;
        }
        c++;
    }

    assert(schedule->dep_id >= 0);
    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    memset(schedule->reqs, 0, sizeof(schedule->reqs));
    (*sched) = &schedule->super;
    return TCCL_OK;
}
