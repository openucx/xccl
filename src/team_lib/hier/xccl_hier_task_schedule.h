/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef XCCL_HIER_TASK_SCHEDULE_H
#define XCCL_HIER_TASK_SCHEDULE_H

#include <xccl_schedule.h>
#include "xccl_team_lib.h"
#include "xccl_hier_team.h"

typedef struct xccl_coll_args xccl_coll_args_t;

typedef struct xccl_hier_task {
    ucc_coll_task_t     super;
    xccl_coll_op_args_t xccl_coll;
    xccl_hier_pair_t    *pair;
    xccl_coll_req_h     req;
} xccl_hier_task_t;

typedef struct xccl_seq_schedule {
    ucc_schedule_t     super;
    xccl_tl_coll_req_t req;
    int                dep;
    xccl_hier_task_t   *tasks;
} xccl_seq_schedule_t;

xccl_status_t build_allreduce_task_schedule(xccl_hier_team_t *team, xccl_coll_op_args_t coll,
                                            xccl_hier_allreduce_spec_t spec, xccl_seq_schedule_t **sched);

#endif
