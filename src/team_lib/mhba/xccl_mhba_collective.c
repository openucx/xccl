/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <xccl_mhba_collective.h>

xccl_status_t
xccl_mhba_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_mhba_coll_req_t **request,
                               xccl_mhba_team_t *team)
{
    xccl_mhba_team_t *mhba_team       = ucs_derived_of(team, xccl_mhba_team_t);
    *request = (xccl_mhba_coll_req_t*)malloc(sizeof(xccl_mhba_coll_req_t));
    if (*request == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    memcpy(&((*request)->args), coll_args, sizeof(xccl_coll_op_args_t));
    (*request)->team      = mhba_team;
    (*request)->super.lib = &xccl_team_lib_mhba.super;
    return XCCL_OK;
}

static void xccl_mhba_reg_fanin_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    void *my_umr_data = team->node.my_umr_data;

    xccl_mhba_info("fanin start");
    /* start task if completion event received */
    task->state = XCCL_TASK_STATE_INPROGRESS;

    /* Everybody register their sbuf/rbuf and put the data into shm umr data */
    /* .... */

    /* Start fanin */
    if (XCCL_OK == xccl_mhba_node_fanin(team, request->seq_num, request->asr_rank)) {
        xccl_mhba_info("fanin complete");
        task->state = XCCL_TASK_STATE_COMPLETED;
        xccl_event_manager_notify(&task->em, XCCL_EVENT_COMPLETED);
    } else {
        xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    }
}

xccl_status_t xccl_mhba_reg_fanin_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    assert(team->node.sbgp->group_rank == request->asr_rank);
    if (XCCL_OK == xccl_mhba_node_fanin(team, request->seq_num, request->asr_rank)) {
        xccl_mhba_info("fanin complete");
        task->state = XCCL_TASK_STATE_COMPLETED;
    }
    return XCCL_OK;
}

static void xccl_mhba_fanout_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    void *my_umr_data = team->node.my_umr_data;
    xccl_mhba_info("fanout start");
    /* start task if completion event received */
    task->state = XCCL_TASK_STATE_INPROGRESS;

    /* Start fanin */
    if (XCCL_OK == xccl_mhba_node_fanout(team, request->seq_num, request->asr_rank)) {
        task->state = XCCL_TASK_STATE_COMPLETED;

        /*Cleanup alg resources - all done */
        xccl_mhba_info("Algorithm completion");
        xccl_event_manager_notify(&task->em, XCCL_EVENT_COMPLETED);
    } else {
        xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    }
}

xccl_status_t xccl_mhba_fanout_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    assert(team->node.sbgp->group_rank != request->asr_rank);
    if (XCCL_OK == xccl_mhba_node_fanout(team, request->seq_num, request->asr_rank)) {
        task->state = XCCL_TASK_STATE_COMPLETED;
        /*Cleanup alg resources - all done */
        xccl_mhba_info("Algorithm completion");
    }
    return XCCL_OK;
}

static void xccl_mhba_transpose_start(xccl_coll_task_t *task) {
    xccl_mhba_info("tranpose start");
    task->state = XCCL_TASK_STATE_INPROGRESS;
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
}
xccl_status_t xccl_mhba_transpose_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    task->state = XCCL_TASK_STATE_COMPLETED;
    return XCCL_OK;
}

static void xccl_mhba_asr_barrier_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;

    xccl_mhba_info("asr barrier start");
    task->state = XCCL_TASK_STATE_INPROGRESS;
    xccl_coll_op_args_t coll = {
        .coll_type = XCCL_BARRIER,
        .alg.set_by_user = 0,
    };
    team->net.ucx_team->ctx->lib->collective_init(&coll, &request->barrier_req,
                                                  team->net.ucx_team);
    team->net.ucx_team->ctx->lib->collective_post(request->barrier_req);
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
}

xccl_status_t xccl_mhba_asr_barrier_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;

    if (XCCL_OK == team->net.ucx_team->ctx->lib->collective_test(request->barrier_req)) {
        team->net.ucx_team->ctx->lib->collective_finalize(request->barrier_req);
        task->state = XCCL_TASK_STATE_COMPLETED;
    }

    return XCCL_OK;
}

static void xccl_mhba_send_blocks_start(xccl_coll_task_t *task) {
    xccl_mhba_info("send blocks start");
    task->state = XCCL_TASK_STATE_INPROGRESS;
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
}
xccl_status_t xccl_mhba_send_blocks_progress(xccl_coll_task_t *task) {
    task->state = XCCL_TASK_STATE_COMPLETED;
    return XCCL_OK;
}

static void xccl_mhba_wait_blocks_start(xccl_coll_task_t *task) {
    xccl_mhba_info("wait blocks start");
    task->state = XCCL_TASK_STATE_INPROGRESS;
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
}
xccl_status_t xccl_mhba_wait_blocks_progress(xccl_coll_task_t *task) {
    task->state = XCCL_TASK_STATE_COMPLETED;
    return XCCL_OK;
}

xccl_status_t
xccl_mhba_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_mhba_coll_req_t *request,
                        xccl_mhba_team_t *team)
{
    int asr_rank = 0; // TODO select?
    int is_asr = (team->node.sbgp->group_rank == asr_rank);
    int n_tasks = (!is_asr) ? 2 : 6;
    int i;
    xccl_schedule_init(&request->schedule, team->super.ctx);
    request->asr_rank = asr_rank;
    assert(asr_rank < team->node.sbgp->group_size);
    request->tasks = (xccl_mhba_task_t*)malloc(sizeof(xccl_mhba_task_t)*n_tasks);
    request->seq_num = team->sequence_number;
    team->sequence_number++;
    for (i = 0; i < n_tasks; i++) {
        request->tasks[i].req = request;
        xccl_coll_task_init(&request->tasks[i].super);
        if (i > 0) {
            xccl_event_manager_subscribe(&request->tasks[i - 1].super.em, XCCL_EVENT_COMPLETED,
                                        &request->tasks[i].super);
        } else {
            //i == 0
            request->tasks[i].super.handlers[XCCL_EVENT_SCHEDULE_STARTED] = xccl_mhba_reg_fanin_start;
            request->tasks[i].super.progress = xccl_mhba_reg_fanin_progress;
            request->tasks[i].super.handlers[XCCL_EVENT_COMPLETED] = NULL;

            xccl_event_manager_subscribe(&request->schedule.super.em, XCCL_EVENT_SCHEDULE_STARTED,
                                        &request->tasks[i].super);
        }
        xccl_schedule_add_task(&request->schedule, &request->tasks[i].super);
    }
    if (!is_asr) {
        request->tasks[1].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_fanout_start;
        request->tasks[1].super.progress = xccl_mhba_fanout_progress;
    } else {
        request->tasks[1].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_transpose_start;
        request->tasks[1].super.progress = xccl_mhba_transpose_progress;

        request->tasks[2].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_asr_barrier_start;
        request->tasks[2].super.progress = xccl_mhba_asr_barrier_progress;

        request->tasks[3].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_send_blocks_start;
        request->tasks[3].super.progress = xccl_mhba_send_blocks_progress;

        request->tasks[4].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_wait_blocks_start;
        request->tasks[4].super.progress = xccl_mhba_wait_blocks_progress;

        request->tasks[5].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_fanout_start;
        request->tasks[5].super.progress = xccl_mhba_fanout_progress;
    }

    return XCCL_OK;
}
