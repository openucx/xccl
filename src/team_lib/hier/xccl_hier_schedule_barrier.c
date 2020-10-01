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

xccl_status_t build_barrier_schedule(xccl_hier_team_t *team,
                                     xccl_hier_barrier_spec_t spec, coll_schedule_t **sched)
{
    int have_node_leaders_group     = (team->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group           = (team->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group   = (team->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);
    int have_node_group             = (team->sbgps[SBGP_NODE].status == SBGP_ENABLED);

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

    xccl_coll_op_args_t coll = {
        .coll_type       = XCCL_BARRIER,
        .alg.set_by_user = 0,
    };

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = XCCL_BARRIER;
            schedule->super.args[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_FANIN;
            schedule->super.args[c].xccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket];
        c++;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = XCCL_BARRIER;
            schedule->super.args[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_FANIN;
            schedule->super.args[c].xccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
    }

    if (team->no_socket && have_node_group) {
        assert(c == 0);
        /* !hava_socket_group && ! have_socket_leaders_group */
        if (top_sbgp == SBGP_NODE) {
            coll.coll_type = XCCL_BARRIER;
            schedule->super.args[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_FANIN;
            schedule->super.args[c].xccl_coll = coll;
        }
        schedule->super.args[c].pair = team->pairs[spec.pairs.node];
        c++;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = XCCL_BARRIER;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.node_leaders];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = XCCL_FANOUT;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = XCCL_FANOUT;
        schedule->super.args[c].xccl_coll = coll;
        schedule->super.args[c].pair = team->pairs[spec.pairs.socket];
        c++;
    }

    if (team->no_socket && have_node_group  && top_sbgp != SBGP_NODE) {
        coll.coll_type = XCCL_FANOUT;
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
