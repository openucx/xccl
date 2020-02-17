/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_SCHEDULE_H
#define MCCL_SCHEDULE_H
#include <unistd.h>
#include "mccl_core.h"

#define MAX_COLL_SCHEDULE_LENGTH 8


#define MAX_BCOL_COLL_DATA 512

typedef struct coll_schedule_t coll_schedule_t;
typedef struct mccl_comm_t mccl_comm_t;

typedef enum {
    MCCL_COLL_SCHED_SEQ,
    MCCL_COLL_SCHED_SINGLE_DEP,
} coll_schedule_type_t;

typedef struct mccl_coll_args {
    tccl_coll_op_args_t tccl_coll;
    mccl_team_t *team;
} mccl_coll_args_t;

typedef struct coll_schedule_t {
    int type;
    int n_colls;
    int n_completed_colls;
    mccl_comm_t *comm;
    mccl_coll_args_t args[MAX_COLL_SCHEDULE_LENGTH];
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

void build_allreduce_schedule_3lvl(mccl_comm_t *comm, coll_schedule_t **schedule,
                                   coll_schedule_type_t sched_type,
                                   int count, tccl_dt_t dtype, tccl_op_t op, void *sbuf, void *rbuf,
                                   int socket_teamtype, int socket_leaders_teamtype,
                                   int node_leaders_teamtype);
void build_barrier_schedule_3lvl(mccl_comm_t *comm, coll_schedule_t **schedule,
                                 coll_schedule_type_t sched_type,
                                 int socket_teamtype, int socket_leaders_teamtype,
                                 int node_leaders_teamtype);
void build_bcast_schedule_3lvl(mccl_comm_t *comm, coll_schedule_t **sched,
                               void *buf, int count, tccl_dt_t dtype, int root,
                               int node_leaders_teamtype);

int coll_schedule_progress(coll_schedule_t *schedule);

#endif
