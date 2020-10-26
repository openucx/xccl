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
#include "xccl_hier_task_schedule.h"

xccl_status_t build_barrier_task_schedule(xccl_hier_team_t *team,
                                          xccl_coll_op_args_t coll,
                                          xccl_hier_barrier_spec_t spec,
                                          xccl_seq_schedule_t **sched) {
    int have_node_leaders_group   = (team->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group         = (team->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (team->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);
    int have_node_group           = (team->sbgps[SBGP_NODE].status == SBGP_ENABLED);

    int node_leaders_group_exists   = (team->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists         = (team->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (team->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    int node_group_exists           = (team->sbgps[SBGP_NODE].status != SBGP_NOT_EXISTS);

    sbgp_type_t top_sbgp;
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

    xccl_seq_schedule_t *schedule = (xccl_seq_schedule_t *) malloc(sizeof(*schedule));
    if (schedule == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    schedule->tasks = (xccl_hier_task_t *) malloc(8 * sizeof(xccl_hier_task_t));
    schedule->dep = 0;
    coll.alg.set_by_user = 0;

    int i, c = 0;

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = XCCL_BARRIER;
            schedule->tasks[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_FANIN;
            schedule->tasks[c].xccl_coll = coll;
        }
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket];
        c++;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = XCCL_BARRIER;
            schedule->tasks[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_FANIN;
            schedule->tasks[c].xccl_coll = coll;
        }
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
    }

    if (team->no_socket && have_node_group) {
        assert(c == 0);
        /* !have_socket_group && ! have_socket_leaders_group */
        if (top_sbgp == SBGP_NODE) {
            coll.coll_type = XCCL_BARRIER;
            schedule->tasks[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_FANIN;
            schedule->tasks[c].xccl_coll = coll;
        }
        schedule->tasks[c].pair = team->pairs[spec.pairs.node];
        c++;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = XCCL_BARRIER;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.node_leaders];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = XCCL_FANOUT;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
    }

    if (have_socket_group && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = XCCL_FANOUT;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket];
        c++;
    }

    if (team->no_socket && have_node_group  && top_sbgp != SBGP_NODE) {
        coll.coll_type = XCCL_FANOUT;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.node];
        c++;
    }

    xccl_schedule_init(&schedule->super, team->super.ctx);
    for (i = 0; i < c; i++) {
        xccl_coll_task_init(&schedule->tasks[i].super);
        schedule->tasks[i].super.progress = hier_task_progress_handler;
        schedule->tasks[i].super.handlers[XCCL_EVENT_COMPLETED] = hier_task_completed_handler;
        schedule->tasks[i].req     = NULL;
        schedule->tasks[i].scratch = NULL;
        if (i > 0) {
            xccl_event_manager_subscribe(&schedule->tasks[i - 1].super.em, XCCL_EVENT_COMPLETED,
                                        &schedule->tasks[i].super);
        } else {
            //i == 0
            schedule->tasks[i].super.handlers[XCCL_EVENT_SCHEDULE_STARTED] = hier_task_completed_handler;
            xccl_event_manager_subscribe(&schedule->super.super.em, XCCL_EVENT_SCHEDULE_STARTED,
                                        &schedule->tasks[i].super);
        }
        xccl_schedule_add_task(&schedule->super, &schedule->tasks[i].super);
    }

    (*sched) = schedule;
    return XCCL_OK;
}
