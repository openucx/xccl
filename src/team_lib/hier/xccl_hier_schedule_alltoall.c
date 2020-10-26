/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include "xccl_hier_schedule.h"
#include "xccl_hier_task_schedule.h"

xccl_status_t build_alltoall_task_schedule(xccl_hier_team_t *team,
                                           xccl_coll_op_args_t coll,
                                           xccl_hier_alltoall_spec_t spec,
                                           xccl_seq_schedule_t **sched)
{
    xccl_seq_schedule_t *schedule;
    int                 i;
    int                 n_tasks;

    assert(team->sbgps[SBGP_FLAT].status == SBGP_ENABLED);

    schedule = (xccl_seq_schedule_t *)malloc(sizeof(xccl_seq_schedule_t));
    if (schedule == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    schedule->dep   = 0;
    schedule->tasks = (xccl_hier_task_t*)malloc(2*sizeof(xccl_hier_task_t));
    if (schedule->tasks == NULL) {
        free(schedule);
        return XCCL_ERR_NO_MEMORY;
    }

    if (team->sbgps[SBGP_NODE].status != SBGP_ENABLED) {
        coll.alg.set_by_user = 0;
        schedule->tasks[0].xccl_coll = coll;
        schedule->tasks[0].pair      = team->pairs[spec.pairs.flat];
        schedule->tasks[0].scratch   = NULL;
        n_tasks                      = 1;
    } else if (team->sbgps[SBGP_NODE].group_size == team->sbgps[SBGP_FLAT].group_size) {
        coll.alg.set_by_user = 0;
        schedule->tasks[0].xccl_coll = coll;
        schedule->tasks[0].pair      = team->pairs[spec.pairs.node];
        schedule->tasks[0].scratch   = NULL;
        n_tasks                      = 1;
    } else {
        int data_unit_length = coll.buffer_info.len;
        int *scratch_node, *scratch_flat;
        int *counts_node, *offset_node, *counts_flat, *offset_flat;
        xccl_coll_op_args_t coll_op;

        scratch_node = (int*)malloc(2*team->sbgps[SBGP_NODE].group_size*sizeof(int));
        scratch_flat = (int*)malloc(2*team->sbgps[SBGP_FLAT].group_size*sizeof(int));
        if ((scratch_node == NULL) || (scratch_flat == NULL)) {
            free(schedule->tasks);
            free(schedule);
            free(scratch_node);
            free(scratch_flat);
            return XCCL_ERR_NO_MEMORY;
        }

        counts_node = scratch_node;
        offset_node = scratch_node + team->sbgps[SBGP_NODE].group_size;
        counts_flat = scratch_flat;
        offset_flat = scratch_flat + team->sbgps[SBGP_FLAT].group_size;

        coll_op.coll_type                     = XCCL_ALLTOALLV;
        coll_op.buffer_info.src_buffer        = coll.buffer_info.src_buffer;
        coll_op.buffer_info.src_datatype      = XCCL_DT_INT8;
        coll_op.buffer_info.dst_buffer        = coll.buffer_info.dst_buffer;
        coll_op.buffer_info.dst_datatype      = XCCL_DT_INT8;
        coll_op.alg.set_by_user               = 0;
        n_tasks                               = 2;

        for(i = 0; i < team->sbgps[SBGP_FLAT].group_size; i++) {
            counts_flat[i] = data_unit_length;
            offset_flat[i] = team->sbgps[SBGP_FLAT].rank_map[i]*data_unit_length;
        }

        for(i = 0; i < team->sbgps[SBGP_NODE].group_size; i++) {
            counts_node[i] = data_unit_length;
            offset_node[i] = team->sbgps[SBGP_NODE].rank_map[i]*data_unit_length;
            counts_flat[team->sbgps[SBGP_NODE].rank_map[i]] = 0;
            offset_flat[team->sbgps[SBGP_NODE].rank_map[i]] = 0;
        }

/*initialize node sbgp task*/
        coll_op.buffer_info.src_counts        = counts_node;
        coll_op.buffer_info.src_displacements = offset_node;
        coll_op.buffer_info.dst_counts        = counts_node;
        coll_op.buffer_info.dst_displacements = offset_node;
        schedule->tasks[0].xccl_coll          = coll_op;
        schedule->tasks[0].pair               = team->pairs[spec.pairs.node];
        schedule->tasks[0].scratch            = scratch_node;

/*initialize flat sbgp task*/
        coll_op.buffer_info.src_counts        = counts_flat;
        coll_op.buffer_info.src_displacements = offset_flat;
        coll_op.buffer_info.dst_counts        = counts_flat;
        coll_op.buffer_info.dst_displacements = offset_flat;
        schedule->tasks[1].xccl_coll          = coll_op;
        schedule->tasks[1].pair               = team->pairs[spec.pairs.flat];
        schedule->tasks[1].scratch            = scratch_flat;
    }

    xccl_schedule_init(&schedule->super, team->super.ctx);
    for(i = 0; i < n_tasks; i++) {
        xccl_coll_task_init(&schedule->tasks[i].super);
        schedule->tasks[i].super.progress = hier_task_progress_handler;
        schedule->tasks[i].super.handlers[XCCL_EVENT_COMPLETED]        = NULL;
        schedule->tasks[i].super.handlers[XCCL_EVENT_SCHEDULE_STARTED] = hier_task_completed_handler;
        schedule->tasks[i].req = NULL;
        xccl_schedule_add_task(&schedule->super, &schedule->tasks[i].super);
        xccl_event_manager_subscribe(&schedule->super.super.em,
                                     XCCL_EVENT_SCHEDULE_STARTED,
                                     &schedule->tasks[i].super);
    }
    (*sched) = schedule;
    return XCCL_OK;
}
            