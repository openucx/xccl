/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef XCCL_HIER_SCHEDULE_H
#define XCCL_HIER_SCHEDULE_H
#include <unistd.h>
#include "xccl_team_lib.h"
#include "xccl_hier_team.h"
#define MAX_COLL_SCHEDULE_LENGTH 8

typedef struct coll_schedule_t coll_schedule_t;
typedef struct xccl_hier_team xccl_hier_team_t;
typedef struct xccl_hier_pair xccl_hier_pair_t;
typedef struct coll_schedule_fragmented coll_schedule_fragmented_t;
typedef xccl_status_t (*coll_schedule_progress_fn_t)(coll_schedule_t *schedule);

typedef enum {
    XCCL_COLL_SCHED_SEQ,
    XCCL_COLL_SCHED_SINGLE_DEP,
    XCCL_COLL_SCHED_FRAGMENTED,
} coll_schedule_type_t;

typedef struct xccl_coll_args {
    xccl_coll_op_args_t xccl_coll;
    xccl_hier_pair_t   *pair;
} xccl_coll_args_t;

typedef struct coll_schedule_t {
    xccl_coll_req_t             super;
    int                         type;
    xccl_status_t               status;
    xccl_hier_team_t           *hier_team;
    coll_schedule_progress_fn_t progress;
} coll_schedule_t;

typedef struct coll_schedule_1frag_t {
    coll_schedule_t             super;
    int                         n_colls;
    int                         n_completed_colls;
    coll_schedule_fragmented_t *fs;
    int                         frag_id;
    xccl_coll_args_t            args[MAX_COLL_SCHEDULE_LENGTH];
} coll_schedule_1frag_t;

typedef struct coll_schedule_sequential {
    coll_schedule_1frag_t super;
    xccl_coll_req_h       req;
} coll_schedule_sequential_t;

typedef struct coll_schedule_single_dep {
    coll_schedule_1frag_t super;
    xccl_coll_req_h       reqs[MAX_COLL_SCHEDULE_LENGTH];
    uint8_t               dep_id;
    uint8_t               dep_satisfied;
} coll_schedule_single_dep_t;

typedef enum {
    COLL_SCHEDULE_FRAG_ON_BYTE,
    COLL_SCHEDULE_FRAG_ON_DTYPE,
} coll_schedule_frag_type_t;

typedef struct coll_schedule_fragmented {
    coll_schedule_t           super;
    int                       n_frags;
    int                       n_frags_launched;
    int                       n_frags_completed;
    int                       pipeline_depth;
    int                       ordered;
    coll_schedule_frag_type_t frag_type;
    xccl_coll_buffer_info_t   binfo;
    int                       level_started_frag_num[MAX_COLL_SCHEDULE_LENGTH];
    union {
        coll_schedule_1frag_t **frags;
        coll_schedule_1frag_t *frag;
    };
} coll_schedule_fragmented_t;

typedef struct xccl_hier_schedule_pairs_t {
    xccl_hier_pair_type_t  socket;
    xccl_hier_pair_type_t  socket_leaders;
    xccl_hier_pair_type_t  node_leaders;
} xccl_hier_schedule_pairs_t;

typedef struct xccl_hier_bcast_spec {
    int                        use_sm_fanout_get;
    xccl_hier_schedule_pairs_t pairs;
    xccl_mem_h                 socket_memh;
    xccl_mem_h                 socket_leaders_memh;
} xccl_hier_bcast_spec_t;

typedef struct xccl_hier_allreduce_spec {
    xccl_hier_schedule_pairs_t pairs;
} xccl_hier_allreduce_spec_t;

typedef struct xccl_hier_barrier_spec {
    xccl_hier_schedule_pairs_t pairs;
} xccl_hier_barrier_spec_t;

xccl_status_t build_allreduce_schedule(xccl_hier_team_t *team, xccl_coll_op_args_t coll,
                                       xccl_hier_allreduce_spec_t spec, coll_schedule_t **schedule);
xccl_status_t build_barrier_schedule(xccl_hier_team_t *team, xccl_hier_barrier_spec_t, 
                                     coll_schedule_t **schedule);
xccl_status_t build_bcast_schedule(xccl_hier_team_t *team, xccl_coll_op_args_t coll,
                                   xccl_hier_bcast_spec_t spec, coll_schedule_t **sched);
xccl_status_t build_bcast_schedule_sm_get(xccl_hier_team_t *comm, coll_schedule_t **sched,
                                          xccl_coll_op_args_t coll);
xccl_status_t coll_schedule_progress_sequential(coll_schedule_t *schedule);
xccl_status_t coll_schedule_progress_single_dep(coll_schedule_t *schedule);
xccl_status_t make_fragmented_schedule(coll_schedule_t *in_sched, coll_schedule_t **frag_sched,
                                       xccl_coll_buffer_info_t binfo,
                                       size_t frag_thresh, int ordered, int pipeline_depth);

static inline
xccl_status_t coll_schedule_progress(coll_schedule_t *schedule) {
    return schedule->progress(schedule);
}
#endif
