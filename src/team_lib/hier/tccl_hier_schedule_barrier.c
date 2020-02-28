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

tccl_status_t build_barrier_schedule_3lvl(tccl_hier_team_t *team, coll_schedule_t **sched,
                                          int socket_pair, int socket_leaders_pair,
                                          int node_leaders_pair)
{
    int have_node_leaders_group = (team->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group = (team->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (team->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);

    int node_leaders_group_exists = (team->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists = (team->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (team->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    sbgp_type_t top_sbgp = node_leaders_group_exists ? SBGP_NODE_LEADERS :
        (socket_leaders_group_exists ? SBGP_SOCKET_LEADERS : SBGP_SOCKET);

    coll_schedule_sequential_t *schedule = (coll_schedule_sequential_t *)malloc(sizeof(*schedule));
    schedule->super.hier_team = team;
    schedule->super.type = TCCL_COLL_SCHED_SEQ;
    int c = 0;

    tccl_coll_op_args_t coll = {
        .coll_type       = TCCL_BARRIER,
        .alg.set_by_user = 0,
    };

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = TCCL_BARRIER;
            schedule->super.args[c].tccl_coll = coll;
        } else {
            coll.coll_type = TCCL_FANIN;
            schedule->super.args[c].tccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[socket_pair];
        c++;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = TCCL_BARRIER;
            schedule->super.args[c].tccl_coll = coll;
        } else {
            coll.coll_type = TCCL_FANIN;
            schedule->super.args[c].tccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[socket_leaders_pair];
        c++;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = TCCL_BARRIER;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[node_leaders_pair];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = TCCL_FANOUT;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[socket_leaders_pair];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = TCCL_FANOUT;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[socket_pair];
        c++;
    }
    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    schedule->req = NULL;

    (*sched) = &schedule->super;
    return TCCL_OK;
}
