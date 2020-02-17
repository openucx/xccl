/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>

#include "schedule.h"
#include "mccl_team.h"
#include "mccl_inline.h"
static inline int
root_at_socket(int root, sbgp_t *sbgp) {
    int i;
    int socket_root_rank = -1;
    for (i=0; i<sbgp->group_size; i++) {
        if (root == sbgp->mccl_rank_map[i]) {
            socket_root_rank = i;
            break;
        }
    }
    return socket_root_rank;
}

void build_bcast_schedule_3lvl(mccl_comm_t *comm, coll_schedule_t **sched,
                               void *buf, int count, tccl_dt_t dtype, int root,
                               int node_leaders_teamtype) {
    int have_node_leaders_group = (comm->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group = (comm->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (comm->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);

    int node_leaders_group_exists = (comm->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists = (comm->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (comm->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    sbgp_type_t top_sbgp = node_leaders_group_exists ? SBGP_NODE_LEADERS :
        (socket_leaders_group_exists ? SBGP_SOCKET_LEADERS : SBGP_SOCKET);

    int rank = comm->config.comm_rank;
    int wroot = mccl_comm_rank2world(comm, root);
    int root_on_local_node = is_rank_on_local_node(root, comm);
    int root_on_local_socket = root_on_local_node &&
        is_rank_on_local_socket(root, comm);
    mccl_context_t *ctx = comm->config.mccl_ctx;

    coll_schedule_single_dep_t *schedule = (coll_schedule_single_dep_t *)malloc(sizeof(*schedule));
    schedule->super.comm = comm;
    schedule->super.type = MCCL_COLL_SCHED_SINGLE_DEP;
    int c = 0;

    int sock_leaders_team = MCCL_TEAM_SOCKET_LEADERS_UCX;
    int sock_team = MCCL_TEAM_SOCKET_UCX;

    tccl_coll_op_args_t coll = {
        .coll_type = TCCL_BCAST,
        .buffer_info = {
            .src_buffer = buf,
            .dst_buffer = buf,
            .len        = count*tccl_dt_size(dtype),
        },
        .root            = 0,
        .alg.set_by_user = 0,
        .tag             = 123, //todo
    };

    if (coll.buffer_info.len <= 2048 && ctx->libs[TCCL_LIB_SHMSEG].enabled) {
        sock_leaders_team = MCCL_TEAM_SOCKET_LEADERS_SHMSEG;
        sock_team = MCCL_TEAM_SOCKET_SHMSEG;
    }

    schedule->dep_id = -1;
    if (rank == root) {
        schedule->dep_id = 0;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.root = ctx->procs[wroot].node_id;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[node_leaders_teamtype];
        if (coll.root != comm->sbgps[SBGP_NODE_LEADERS].group_rank) {
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
        schedule->super.args[c].team = comm->teams[sock_leaders_team];
        if (coll.root != comm->sbgps[SBGP_SOCKET_LEADERS].group_rank) {
            schedule->dep_id = c;
        }
        c++;
    }
    if (have_socket_group) {
        if (root_on_local_socket) {
            coll.root = root_at_socket(root, &comm->sbgps[SBGP_SOCKET]);
        } else {
            coll.root = 0;
        }
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[sock_team];
        if (coll.root != comm->sbgps[SBGP_SOCKET].group_rank) {
            schedule->dep_id = c;
        }
        c++;
    }

    assert(schedule->dep_id >= 0);

    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    memset(schedule->reqs, 0, sizeof(schedule->reqs));
    (*sched) = &schedule->super;
}
