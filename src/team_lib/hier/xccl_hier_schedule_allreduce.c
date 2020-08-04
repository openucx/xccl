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

xccl_status_t build_allreduce_schedule(xccl_hier_team_t *team, xccl_coll_op_args_t coll,
                                       xccl_hier_allreduce_spec_t spec, coll_schedule_t **sched)
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
    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    schedule->req = NULL;
    (*sched) = &schedule->super.super;
    return XCCL_OK;
}

ucc_status_t hier_seq_task_progress_handler(ucc_coll_task_t *task)
{
    const int n_polls = 10;
    xccl_status_t status;
    int i;

    xccl_hier_task_t *self = (xccl_hier_task_t*)task;
    for (i = 0; (i < n_polls) && (task->state == UCC_TASK_STATE_INPROGRESS); i++) {
        if (!self->req) {
            status = xccl_collective_init(&self->xccl_coll, &self->req, self->pair->team);
            status = xccl_collective_post(self->req);
        }
        status = xccl_collective_test(self->req);
        if (status == XCCL_OK) {
            xccl_collective_finalize(self->req);
            ucc_event_manager_notify(&task->em, UCC_EVENT_COMPLETED);
            task->state = UCC_TASK_STATE_COMPLETED;
        }
    }
    return task->state == UCC_TASK_STATE_COMPLETED ? UCC_OK : UCC_INPROGRESS;
}

void hier_seq_task_completed_handler(ucc_coll_task_t *task)
{
    /* start task if completion event received */
    task->state = UCC_TASK_STATE_INPROGRESS;
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
}

xccl_status_t build_allreduce_task_schedule(xccl_hier_team_t *team, xccl_coll_op_args_t coll,
                                            xccl_hier_allreduce_spec_t spec, xccl_seq_schedule_t **sched)
{
    int have_node_leaders_group     = (team->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group           = (team->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group   = (team->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);
    int node_leaders_group_exists   = (team->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists         = (team->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (team->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    sbgp_type_t top_sbgp            = node_leaders_group_exists ? SBGP_NODE_LEADERS :
                                      (socket_leaders_group_exists ? SBGP_SOCKET_LEADERS : SBGP_SOCKET);
    int i,c = 0;

    xccl_seq_schedule_t *schedule = (xccl_seq_schedule_t *)malloc(sizeof(*schedule));
    if (schedule == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    schedule->tasks = (xccl_hier_task_t*)malloc(8*sizeof(xccl_hier_task_t));
    schedule->dep = 0;
    coll.root = 0;
    coll.alg.set_by_user = 0;

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = XCCL_ALLREDUCE;
            schedule->tasks[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_REDUCE;
            schedule->tasks[c].xccl_coll = coll;
        }
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket];
        c++;
        coll.buffer_info.src_buffer = coll.buffer_info.dst_buffer;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = XCCL_ALLREDUCE;
            schedule->tasks[c].xccl_coll = coll;
        } else {
            coll.coll_type = XCCL_REDUCE;
            schedule->tasks[c].xccl_coll = coll;
        }
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
        coll.buffer_info.src_buffer = coll.buffer_info.dst_buffer;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = XCCL_ALLREDUCE;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.node_leaders];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = XCCL_BCAST;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket_leaders];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = XCCL_BCAST;
        schedule->tasks[c].xccl_coll = coll;
        schedule->tasks[c].pair = team->pairs[spec.pairs.socket];
        c++;
    }

    ucc_schedule_init(&schedule->super, team->super.ctx);
    for (i = 0; i < c; i++) {
        ucc_coll_task_init(&schedule->tasks[i].super);
        schedule->tasks[i].super.progress  = hier_seq_task_progress_handler;
        schedule->tasks[i].super.handlers[UCC_EVENT_COMPLETED] = hier_seq_task_completed_handler;
        schedule->tasks[i].req = NULL;
        if (i > 0) {
            ucc_event_manager_subscribe(&schedule->tasks[i-1].super.em, UCC_EVENT_COMPLETED, &schedule->tasks[i].super);
        } else {
            //i == 0
            schedule->tasks[i].super.handlers[UCC_EVENT_SCHEDULE_STARTED] = hier_seq_task_completed_handler;
            ucc_event_manager_subscribe(&schedule->super.super.em, UCC_EVENT_SCHEDULE_STARTED,
                                        &schedule->tasks[i].super);
        }
        ucc_schedule_add_task(&schedule->super, &schedule->tasks[i].super);
    }
    (*sched) = schedule;
    return XCCL_OK;
}
