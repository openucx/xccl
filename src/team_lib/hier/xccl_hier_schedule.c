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
#include "xccl_hier_task_schedule.h"
#include "xccl_hier_team.h"

static xccl_tl_coll_req_t xccl_hier_complete_req;
#define REQ_COMPLETE ((xccl_coll_req_h)&xccl_hier_complete_req)

static inline int can_start_level(coll_schedule_frag_t *sched, int level) {
    if (sched->fs && sched->fs->ordered) {
        return sched->frag_id == sched->fs->level_started_frag_num[level] + 1 ? 1 : 0;
    }
    return 1;
}

static inline void update_level_started(coll_schedule_frag_t *sched, int level) {
    if (sched->fs && sched->fs->ordered) {
        assert(sched->fs->level_started_frag_num[level] ==
               sched->frag_id - 1);
        sched->fs->level_started_frag_num[level]++;
    }
}

/* This function should be used to launch collectives on a given hierarchy level
   for the schedules that support fragmentation.
   - This function will check if the collective at a given level CAN be launched right now
   due to possible ordering dependency (if the previous fragment has been launched already
   at the same given hierarchy level).
   - This function also updates the ordering information of the fragmented schedule */
static inline xccl_status_t launch_coll_op(coll_schedule_frag_t *schedule,
                                           int level, xccl_coll_req_h *req)
{
    xccl_coll_args_t *curr_op = &schedule->args[level];
    xccl_status_t status;
    assert(schedule->super.type == XCCL_COLL_SCHED_SINGLE_DEP ||
           schedule->super.type == XCCL_COLL_SCHED_SEQ);
    if (!can_start_level(schedule, level)) {
        return XCCL_ERR_NO_PROGRESS;
    }
    if (XCCL_OK != (status =
                    xccl_collective_init(&curr_op->xccl_coll, req,
                                         curr_op->pair->team))) {
        return status;
    }
    if (XCCL_OK != (status = xccl_collective_post(*req))) {
        return status;
    }
    update_level_started(schedule, level);
    return status;
}

xccl_status_t coll_schedule_progress_sequential(coll_schedule_t *sched)
{
    int i;
    int curr_idx;
    coll_schedule_sequential_t *schedule = ucs_derived_of(sched, coll_schedule_sequential_t);
    int n_colls = schedule->super.n_colls;
    xccl_status_t status;
    const int n_polls = 10;
    for (i=0; i<n_polls && n_colls != schedule->super.n_completed_colls; i++) {
        curr_idx = schedule->super.n_completed_colls;
        if (!schedule->req) {
            if (XCCL_OK != (status  = launch_coll_op(&schedule->super, curr_idx,
                                                     &schedule->req))) {
                return status == XCCL_ERR_NO_PROGRESS ? XCCL_OK : status;
            }
        }
        if (XCCL_OK == xccl_collective_test(schedule->req)) {
            schedule->super.n_completed_colls++;
            xccl_collective_finalize(schedule->req);
            schedule->req = NULL;
            i = 0;
            if (schedule->super.n_completed_colls == n_colls) {
                schedule->super.super.status = XCCL_OK;
            }
        }
    }

    return XCCL_OK;
}

/* Progress schedule with a single main dependency:
   - If the dependency is not satisfied yet:
       - Start it if not started yet
       - Test & progress. Launch ALL other levels of schedule when dependency is completed.
   - If dep_satisfied: test&progress ALL other levels until all are done. */
xccl_status_t coll_schedule_progress_single_dep(coll_schedule_t *sched)
{
    int i, p;
    int curr_idx;
    coll_schedule_single_dep_t *schedule = ucs_derived_of(sched, coll_schedule_single_dep_t);
    int n_colls = schedule->super.n_colls;
    const int n_polls = 10;
    xccl_status_t status;
    if (!schedule->dep_satisfied) {
        curr_idx = schedule->dep_id;
        if (!schedule->reqs[curr_idx]) {
            /* fprintf(stderr,"starting first dep: frag_id %d, n_completed %d, %s, group_rank %d, l_root %d\n", */
            /*         schedule->super.frag_id, */
            /*         schedule->super.n_completed_colls, */
            /*         sbgp_type_str[schedule->super.args[curr_idx].pair->sbgp->type], */
            /*         schedule->super.args[curr_idx].pair->sbgp->group_rank, */
            /*         schedule->super.args[curr_idx].xccl_coll.root); */
            if (XCCL_OK != (status  = launch_coll_op(&schedule->super, curr_idx,
                                                     &schedule->reqs[curr_idx]))) {
                return status == XCCL_ERR_NO_PROGRESS ? XCCL_OK : status;
            }
        }
        for (p=0; p<n_polls; p++) {
            if (XCCL_OK == xccl_collective_test(schedule->reqs[curr_idx])) {
                /* fprintf(stderr, "completed first dep, starting all others: %d\n", */
                /*         n_colls-1); */
                schedule->super.n_completed_colls++;
                if (schedule->super.n_completed_colls == n_colls) {
                    schedule->super.super.status = XCCL_OK;
                }
                xccl_collective_finalize(schedule->reqs[curr_idx]);
                schedule->reqs[curr_idx] = REQ_COMPLETE;
                schedule->dep_satisfied = 1;
                break;
            }
        }
    }

    if (schedule->dep_satisfied) {
        for (p=0; p<n_polls && n_colls != schedule->super.n_completed_colls; p++) {
            for (i=0; i<n_colls; i++) {
                if (!schedule->reqs[i]) {
                    if (XCCL_OK != (status  = launch_coll_op(&schedule->super,
                                                             i, &schedule->reqs[i]))) {
                        if (XCCL_ERR_NO_PROGRESS == status) {
                            continue;
                        } else {
                            return status;
                        }
                    }
                } else if (REQ_COMPLETE != schedule->reqs[i]) {
                    if (XCCL_OK == xccl_collective_test(schedule->reqs[i])) {
                        schedule->super.n_completed_colls++;
                        /* fprintf(stderr, "completed [%d], %s, n_completed %d\n", */
                        /*         i, sbgp_type_str[schedule->super.args[i].pair->sbgp->type], */
                        /*         schedule->super.n_completed_colls); */
                        xccl_collective_finalize(schedule->reqs[i]);
                        schedule->reqs[i] = REQ_COMPLETE;
                        p = 0;
                        if (schedule->super.n_completed_colls == n_colls) {
                            schedule->super.super.status = XCCL_OK;
                        }
                    }
                }
            }
        }
    }
    return XCCL_OK;
}

static inline coll_schedule_frag_t *
get_frag(coll_schedule_fragmented_t *fs, int frag_pipeline_num)
{
    if (fs->pipeline_depth == 1) {
        assert(frag_pipeline_num == 0);
        return fs->frag;
    } else {
        return fs->frags[frag_pipeline_num];
    }
}

static inline void coll_schedule_reset(coll_schedule_t *schedule) {
    if (XCCL_COLL_SCHED_SINGLE_DEP == schedule->type) {
                coll_schedule_single_dep_t *sched =
                    (coll_schedule_single_dep_t *)schedule;
                sched->super.n_completed_colls = 0;
                sched->dep_satisfied = 0;
                memset(sched->reqs, 0, sizeof(sched->reqs));
    }
    if (XCCL_COLL_SCHED_SEQ == schedule->type) {
                coll_schedule_sequential_t *sched =
                    (coll_schedule_sequential_t *)schedule;
                sched->super.n_completed_colls = 0;
                sched->req = NULL;
    }
    schedule->status = XCCL_INPROGRESS;
}

static inline void init_frag(coll_schedule_fragmented_t *fs, int frag_pipeline_num)
{
    coll_schedule_frag_t *frag_sched = get_frag(fs, frag_pipeline_num);
    int frag_num                      = fs->n_frags_launched;
    size_t frag_size                  = fs->binfo.len / fs->n_frags;
    size_t left                       = fs->binfo.len % fs->n_frags;
    size_t offset                     = frag_num * frag_size + left;
    //TODO fragmentation on the dtype boundary
    assert(fs->frag_type == COLL_SCHEDULE_FRAG_ON_BYTE);
    int i;
    if (frag_num == fs->n_frags) {
        /* All frags have been launched. Nothing to do. */
        return;
    }
    if (frag_num < left) {
        frag_size++;
        offset -= left - frag_num;
    }
    assert(frag_sched->super.type == XCCL_COLL_SCHED_SINGLE_DEP ||
           frag_sched->super.type == XCCL_COLL_SCHED_SEQ);
    /* Reset the schedule specific parameter - the same schedule is going to be re-used
       again but with the new buffer info. Should be reset to the init state. */
    coll_schedule_reset(&frag_sched->super);
    frag_sched->frag_id = frag_num;
    /* fprintf(stderr, "init frag %d, total_frags %d, total_len %zd, frag_len %zd, offset %zd\n", */
    /*         frag_num, fs->n_frags, fs->binfo.len, frag_size, offset); */

    /* Set the buffer information.
       FANOUT_GET is a special case, get_info is used instead of buffer_info. */
    for (i=0; i<frag_sched->n_colls; i++) {
        xccl_coll_op_args_t *coll = &frag_sched->args[i].xccl_coll;
        if (coll->coll_type == XCCL_FANOUT_GET) {
            coll->get_info.offset = offset;
            coll->get_info.local_buffer = (void*)((ptrdiff_t)fs->binfo.dst_buffer + offset);
            coll->get_info.len = frag_size;
        } else {
            coll->buffer_info.src_buffer = (void*)((ptrdiff_t)fs->binfo.src_buffer + offset);
            coll->buffer_info.dst_buffer = (void*)((ptrdiff_t)fs->binfo.dst_buffer + offset);
            coll->buffer_info.len = frag_size;
        }
    }
    fs->n_frags_launched++;
}

/* Goes over all the oustanding fragments in the pipeline and calls their
   progress functions. Upon completion of any fragment - init next one */
xccl_status_t coll_schedule_progress_fragmented(coll_schedule_t *sched) {
    coll_schedule_fragmented_t *fs = (coll_schedule_fragmented_t*)sched;
    int i;
    for (i=0; i<fs->pipeline_depth; i++) {
        coll_schedule_frag_t *s = get_frag(fs, i);
        if (XCCL_INPROGRESS == s->super.status) {
            coll_schedule_progress(&s->super);
            if (XCCL_OK == s->super.status) {
                /* fprintf(stderr, "Completed frag %d\n", ((coll_schedule_frag_t*)s)->frag_id); */
                fs->n_frags_completed++;
                init_frag(fs, i);
            }
        }
    }
    if (fs->n_frags_completed == fs->n_frags) {
        fs->super.status = XCCL_OK;
    }
    return XCCL_OK;
}

static inline coll_schedule_t*
schedule_dup(coll_schedule_t *in_schedule) {
    coll_schedule_t *out_schedule;
    size_t sched_size;
    switch (in_schedule->type) {
    case XCCL_COLL_SCHED_SEQ:
        sched_size = sizeof(coll_schedule_sequential_t);
        break;
    case XCCL_COLL_SCHED_SINGLE_DEP:
        sched_size = sizeof(coll_schedule_single_dep_t);
        break;
    case XCCL_COLL_SCHED_FRAGMENTED:
        sched_size = sizeof(coll_schedule_fragmented_t);
        break;
    }
    out_schedule = malloc(sched_size);
    memcpy(out_schedule, in_schedule, sched_size);
    return out_schedule;
}
xccl_status_t make_fragmented_schedule(coll_schedule_t *in_sched, coll_schedule_t **frag_sched,
                                       xccl_coll_buffer_info_t binfo,
                                       size_t frag_thresh, int ordered, int pipeline_depth)
{
    int i;
    coll_schedule_fragmented_t *fs = (coll_schedule_fragmented_t*)malloc(sizeof(*fs));
    fs->super.type        = XCCL_COLL_SCHED_FRAGMENTED;
    fs->super.status      = XCCL_INPROGRESS;
    fs->super.hier_team   = in_sched->hier_team;
    fs->super.progress    = coll_schedule_progress_fragmented;
    fs->n_frags           = (binfo.len + frag_thresh - 1)/frag_thresh;
    fs->n_frags_launched  = 0;
    fs->n_frags_completed = 0;
    fs->binfo             = binfo;
    fs->frag_type         = COLL_SCHEDULE_FRAG_ON_BYTE; //TODO on count, for REDUCTIONS
    fs->pipeline_depth    = pipeline_depth <= fs->n_frags ? pipeline_depth : fs->n_frags;
    fs->ordered           = ordered;
    /*only 2 schedule types support fragmentation so far */
    assert(in_sched->type == XCCL_COLL_SCHED_SINGLE_DEP ||
           in_sched->type == XCCL_COLL_SCHED_SEQ);

    if (pipeline_depth == 1) {
        fs->frag = (coll_schedule_frag_t*)in_sched;
    } else {
        /* Pipeline depth is more than 1. Need to allocate the storage for
           "pipline_depth" number of fragments and duplicate collective schedules.
           The buffer information for those schedules will be set in init_frag */
        fs->frags = malloc(pipeline_depth*sizeof(*fs->frags));
        fs->frags[0] = (coll_schedule_frag_t*)in_sched;
        for (i=1; i<pipeline_depth; i++) {
            fs->frags[i] = (coll_schedule_frag_t*)schedule_dup(in_sched);
        }
    }
    /* Set the "fs" pointer - it will indicate the schedule is part of fragmented collective */
    for (i=0; i<pipeline_depth; i++) {
        get_frag(fs, i)->fs = fs;
        init_frag(fs, i);
    }
    /* Initialize ordering information */
    for (i=0; i<MAX_COLL_SCHEDULE_LENGTH; i++) {
        fs->level_started_frag_num[i] = -1;
    }
    *frag_sched = &fs->super;
    return XCCL_OK;
}


xccl_status_t hier_task_progress_handler(xccl_coll_task_t *task)
{
    const int n_polls = 10;
    xccl_status_t status;
    int i;
    xccl_hier_task_t *self = (xccl_hier_task_t *) task;
    assert(self->req);
    for (i = 0; (i < n_polls) && (task->state == XCCL_TASK_STATE_INPROGRESS); i++) {
        status = xccl_collective_test(self->req);
        if (status == XCCL_OK) {
            xccl_collective_finalize(self->req);
            task->state = XCCL_TASK_STATE_COMPLETED;
        }
    }
    return XCCL_OK;
}

void hier_task_completed_handler(xccl_coll_task_t *task)
{
    xccl_hier_task_t *self = (xccl_hier_task_t *) task;
    /* start task if completion event received */
    task->state = XCCL_TASK_STATE_INPROGRESS;
    assert(NULL == self->req);
    xccl_collective_init(&self->xccl_coll, &self->req, self->pair->team);
    xccl_collective_post(self->req);
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
}
