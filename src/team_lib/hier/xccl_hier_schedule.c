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

xccl_status_t coll_schedule_progress_sequential(coll_schedule_t *sched)
{
    int i;
    int curr_idx;
    xccl_coll_args_t *curr_op;
    coll_schedule_sequential_t *schedule = xccl_derived_of(sched, coll_schedule_sequential_t);
    int n_colls = schedule->super.n_colls;
    const int n_polls = 10;
    for (i=0; i<n_polls && n_colls != schedule->super.n_completed_colls; i++) {
        curr_idx = schedule->super.n_completed_colls;
        curr_op = &schedule->super.args[curr_idx];
        if (!schedule->req) {
            xccl_collective_init(&curr_op->xccl_coll, &schedule->req,
                                 curr_op->pair->team);
            xccl_collective_post(schedule->req);
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

xccl_status_t coll_schedule_progress_single_dep(coll_schedule_t *sched)
{
    int i, p;
    int curr_idx;
    xccl_coll_args_t *curr_op;
    coll_schedule_single_dep_t *schedule = xccl_derived_of(sched, coll_schedule_single_dep_t);
    int n_colls = schedule->super.n_colls;
    const int n_polls = 10;
    if (schedule->dep_id >= 0) {
        curr_idx = schedule->dep_id;
        curr_op = &schedule->super.args[curr_idx];
        if (!schedule->reqs[curr_idx]) {
            /* fprintf(stderr,"starting first dep: n_completed %d, %s, group_rank %d, l_root %d\n", */
            /*         schedule->super.n_completed_colls, */
            /*         sbgp_type_str[curr_op->team->sbgp->type], */
            /*         curr_op->team->sbgp->group_rank, */
            /*         curr_op->xccl_coll.root); */

            xccl_collective_init(&curr_op->xccl_coll, &schedule->reqs[curr_idx],
                                curr_op->pair->team);
            xccl_collective_post(schedule->reqs[curr_idx]);
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
                schedule->reqs[curr_idx] = NULL;
                for (i=0; i<n_colls; i++) {
                    if (i != curr_idx) {
                        curr_op = &schedule->super.args[i];
                        assert(NULL == schedule->reqs[i]);
                        /* fprintf(stderr,"starting [%d]: %s, group_rank %d, l_root %d\n", i, */
                        /*         sbgp_type_str[curr_op->team->sbgp->type], */
                        /*         curr_op->team->sbgp->group_rank, */
                        /*         curr_op->xccl_coll.root); */

                        xccl_collective_init(&curr_op->xccl_coll, &schedule->reqs[i],
                                            curr_op->pair->team);
                        xccl_collective_post(schedule->reqs[i]);
                    }
                }
                schedule->dep_id = -1;
                break;
            }
        }
    }

    if (schedule->dep_id < 0) {
        for (p=0; p<n_polls && n_colls != schedule->super.n_completed_colls; p++) {
            for (i=0; i<n_colls; i++) {
                if (schedule->reqs[i]) {
                    if (XCCL_OK == xccl_collective_test(schedule->reqs[i])) {
                        schedule->super.n_completed_colls++;
                        /* fprintf(stderr, "completed [%d], %s, n_completed %d\n", */
                        /*         i, sbgp_type_str[schedule->super.args[i].team->sbgp->type], */
                        /*         schedule->super.n_completed_colls); */
                        xccl_collective_finalize(schedule->reqs[i]);
                        schedule->reqs[i] = NULL;
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
