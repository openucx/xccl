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

xccl_status_t build_allreduce_schedule(xccl_hier_team_t *team, xccl_coll_op_args_t coll,
                                       xccl_hier_allreduce_spec_t spec, coll_schedule_t **sched)
{
    int have_node_leaders_group   = (team->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group         = (team->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (team->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);
    int have_node_group           = (team->sbgps[SBGP_NODE].status == SBGP_ENABLED);

    int node_leaders_group_exists   = (team->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists         = (team->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (team->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    int node_group_exists           = (team->sbgps[SBGP_NODE].status != SBGP_NOT_EXISTS);

    sbgp_type_t top_sbgp;
    coll_schedule_sequential_t *schedule = (coll_schedule_sequential_t *)malloc(sizeof(*schedule));

    if (node_leaders_group_exists) {
        top_sbgp = SBGP_NODE_LEADERS;
    } else if (socket_leaders_group_exists) {
        top_sbgp = SBGP_SOCKET_LEADERS;
    } else if (socket_group_exists) {
        top_sbgp = SBGP_SOCKET;
    } else {
        assert(node_group_exists);
        top_sbgp = SBGP_NODE;
    }

    schedule->super.super.hier_team = team;
    schedule->super.super.type = XCCL_COLL_SCHED_SEQ;
    schedule->super.super.progress = coll_schedule_progress_sequential;
    schedule->super.super.status = XCCL_INPROGRESS;
    schedule->super.fs = NULL;
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
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket];
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
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
        coll.buffer_info.src_buffer = coll.buffer_info.dst_buffer;
    }

    if (team->no_socket && have_node_group) {
        assert(c == 0);
        /* !hava_socket_group && ! have_socket_leaders_group */
        if (top_sbgp == SBGP_NODE) {
            coll.coll_type = XCCL_ALLREDUCE;
            schedule->super.args[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_REDUCE;
            schedule->super.args[c].xccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[spec.pairs.node];
        c++;
        coll.buffer_info.src_buffer = coll.buffer_info.dst_buffer;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = XCCL_ALLREDUCE;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.node_leaders];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = XCCL_BCAST;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = XCCL_BCAST;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket];
        c++;
    }

    if (team->no_socket && have_node_group  && top_sbgp != SBGP_NODE) {
        coll.coll_type = XCCL_BCAST;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.node];
        c++;
    }

    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    schedule->req = NULL;
    (*sched) = &schedule->super.super;
    return XCCL_OK;
}
