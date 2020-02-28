/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef XCCL_HIER_SCHEDULE_H
#define XCCL_HIER_SCHEDULE_H
#include <unistd.h>
#include "xccl_team_lib.h"

#define MAX_COLL_SCHEDULE_LENGTH 8

typedef struct coll_schedule_t coll_schedule_t;
typedef struct xccl_hier_team xccl_hier_team_t;
typedef struct xccl_hier_pair xccl_hier_pair_t;

typedef enum {
    XCCL_COLL_SCHED_SEQ,
    XCCL_COLL_SCHED_SINGLE_DEP,
} coll_schedule_type_t;

typedef struct xccl_coll_args {
    xccl_coll_op_args_t xccl_coll;
    xccl_hier_pair_t   *pair;
} xccl_coll_args_t;

typedef struct coll_schedule_t {
    xccl_coll_req_t super;
    int type;
    int n_colls;
    int n_completed_colls;
    xccl_hier_team_t *hier_team;
    xccl_coll_args_t  args[MAX_COLL_SCHEDULE_LENGTH];
} coll_schedule_t;

typedef struct coll_schedule_sequential {
    coll_schedule_t super;
    xccl_coll_req_h req;
} coll_schedule_sequential_t;

typedef struct coll_schedule_single_dep {
    coll_schedule_t super;
    xccl_coll_req_h reqs[MAX_COLL_SCHEDULE_LENGTH];
    int dep_id;
} coll_schedule_single_dep_t;

xccl_status_t build_allreduce_schedule_3lvl(xccl_hier_team_t *team, coll_schedule_t **schedule,
                                            xccl_coll_op_args_t coll,
                                            int socket_pair, int socket_leaders_pair,
                                            int node_leaders_pair);
xccl_status_t build_barrier_schedule_3lvl(xccl_hier_team_t *team, coll_schedule_t **schedule,
                                          int socket_pair, int socket_leaders_pair,
                                          int node_leaders_pair);
xccl_status_t build_bcast_schedule_3lvl(xccl_hier_team_t *comm, coll_schedule_t **sched,
                                        xccl_coll_op_args_t coll, int node_leaders_pair);

xccl_status_t coll_schedule_progress(coll_schedule_t *schedule);

#endif
