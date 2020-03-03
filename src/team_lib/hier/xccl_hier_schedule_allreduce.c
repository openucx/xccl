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
#include "xccl_hier_schedule.h"
#include "xccl_hier_team.h"

xccl_status_t build_allreduce_schedule_3lvl(xccl_hier_team_t *team, coll_schedule_t **sched,
                                            xccl_coll_op_args_t coll,
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
    schedule->super.super.hier_team = team;
    schedule->super.super.type = XCCL_COLL_SCHED_SEQ;
    schedule->super.super.progress = coll_schedule_progress_sequential;
    schedule->super.super.status = XCCL_INPROGRESS;
    int c = 0;
    coll.root = 0;
    coll.alg.set_by_user = 0;

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = XCCL_ALLREDUCE;
            schedule->super.args[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_REDUCE;
            schedule->super.args[c].xccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[socket_pair];
        c++;
        coll.buffer_info.src_buffer = coll.buffer_info.dst_buffer;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = XCCL_ALLREDUCE;
            schedule->super.args[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_REDUCE;
            schedule->super.args[c].xccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[socket_leaders_pair];
        c++;
        coll.buffer_info.src_buffer = coll.buffer_info.dst_buffer;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = XCCL_ALLREDUCE;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[node_leaders_pair];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = XCCL_BCAST;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[socket_leaders_pair];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = XCCL_BCAST;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[socket_pair];
        c++;
    }
    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    schedule->req = NULL;
    (*sched) = &schedule->super.super;
    return XCCL_OK;
}
