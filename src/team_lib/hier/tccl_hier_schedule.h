/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef TCCL_HIER_SCHEDULE_H
#define TCCL_HIER_SCHEDULE_H
#include <unistd.h>
#include "tccl_team_lib.h"

#define MAX_COLL_SCHEDULE_LENGTH 8

typedef struct coll_schedule_t coll_schedule_t;
typedef struct tccl_hier_team tccl_hier_team_t;
typedef struct tccl_hier_pair tccl_hier_pair_t;

typedef enum {
    TCCL_COLL_SCHED_SEQ,
    TCCL_COLL_SCHED_SINGLE_DEP,
} coll_schedule_type_t;

typedef struct tccl_coll_args {
    tccl_coll_op_args_t tccl_coll;
    tccl_hier_pair_t   *pair;
} tccl_coll_args_t;

typedef struct coll_schedule_t {
    tccl_coll_req_t super;
    int type;
    int n_colls;
    int n_completed_colls;
    tccl_hier_team_t *hier_team;
    tccl_coll_args_t  args[MAX_COLL_SCHEDULE_LENGTH];
} coll_schedule_t;

typedef struct coll_schedule_sequential {
    coll_schedule_t super;
    tccl_coll_req_h req;
} coll_schedule_sequential_t;

typedef struct coll_schedule_single_dep {
    coll_schedule_t super;
    tccl_coll_req_h reqs[MAX_COLL_SCHEDULE_LENGTH];
    int dep_id;
} coll_schedule_single_dep_t;

tccl_status_t build_allreduce_schedule_3lvl(tccl_hier_team_t *team, coll_schedule_t **schedule,
                                            tccl_coll_op_args_t coll,
                                            int socket_pair, int socket_leaders_pair,
                                            int node_leaders_pair);
tccl_status_t build_barrier_schedule_3lvl(tccl_hier_team_t *team, coll_schedule_t **schedule,
                                          int socket_pair, int socket_leaders_pair,
                                          int node_leaders_pair);
tccl_status_t build_bcast_schedule_3lvl(tccl_hier_team_t *comm, coll_schedule_t **sched,
                                        tccl_coll_op_args_t coll, int node_leaders_pair);

tccl_status_t coll_schedule_progress(coll_schedule_t *schedule);

#endif
