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

/* Base schedule representing 1 fragment of the bigger fragmented collective.
   Any collective schedule that needs fragmentation support should be
   derived from this struct. */
typedef struct coll_schedule_1frag_t {
    coll_schedule_t             super;
    int                         n_colls; /**< Number of collectives in a schedule,ie number of steps/levels. */
    int                         n_completed_colls; /*<< number of completed colls so far */
    coll_schedule_fragmented_t *fs; /**< Pointer to the fragmentation schedule.
                                       Can be NULL for non-fragmented collectives. */
    int                         frag_id; /**< Fragment ID of the current schedule.
                                            Only used if "fs" is set.*/
    xccl_coll_args_t            args[MAX_COLL_SCHEDULE_LENGTH];
} coll_schedule_1frag_t;

/* Simple sequential schedule: all the steps at differente levels of hierarchy
   are executed one after another. Only sinlge "req" is required at a time. */
typedef struct coll_schedule_sequential {
    coll_schedule_1frag_t super;
    xccl_coll_req_h       req;
} coll_schedule_sequential_t;

/* A collective schedule with a single dependency, ie. one step in a schedule should
   be completed first and all other steps/levels can be launched/executed concurrently
   w/o additional dependencies between each other. */
typedef struct coll_schedule_single_dep {
    coll_schedule_1frag_t super;
    xccl_coll_req_h       reqs[MAX_COLL_SCHEDULE_LENGTH];
    uint8_t               dep_id; /**< ID of the first step in a schedule that is a main dependency */
    uint8_t               dep_satisfied;
} coll_schedule_single_dep_t;

typedef enum {
    COLL_SCHEDULE_FRAG_ON_BYTE, /**<Fragmentation is done with byte granularity */
    COLL_SCHEDULE_FRAG_ON_DTYPE, /**<Fragmentation is done on Datatype boundary */
} coll_schedule_frag_type_t;

typedef struct coll_schedule_fragmented {
    coll_schedule_t           super;
    int                       n_frags; /**< Total number of fragments in collective */
    int                       n_frags_launched; /**< Number of frags launched so far */
    int                       n_frags_completed; /**< Number of frags completed */
    int                       pipeline_depth; /**< Number of flrags that can be outstanding at a time */
    int                       ordered; /**< Specifies if the fragments launches has to be ordered.
                                          If flag is set then the coll_schedule_progress_fragmented will
                                          make sure that the ordering of the fragments
                                          (at the same hierarchy level) is kept and the corresponding
                                          xccl_collective_init calls appear IN ORDER from the lower level
                                          teams' perspective */
    coll_schedule_frag_type_t frag_type;
    xccl_coll_buffer_info_t   binfo; /**< Replica of the origin buffer_info.
                                        Used to initialize per-fragment buffer info */
    int                       level_started_frag_num[MAX_COLL_SCHEDULE_LENGTH]; /**< At each hierarchy level
                                                       this array keeps the ID of the last fragment launched
                                                       at this level. This array is used if "ordered" flag is
                                                       set to guarantee the proper ordering. */
    union {
        coll_schedule_1frag_t **frags;/**< Array of fragments pointers.
                                         The size of the array equals "pipeline_depth". */
        coll_schedule_1frag_t *frag; /**< Short cut for the case of pipeline_depth = 1 -
                                        no need for speciala array allocation */
    };
} coll_schedule_fragmented_t;

/* Descriptor of the hierarchy pairs (ie TLS) that should be used at different levels */
typedef struct xccl_hier_schedule_pairs_t {
    xccl_hier_pair_type_t  socket;
    xccl_hier_pair_type_t  socket_leaders;
    xccl_hier_pair_type_t  node_leaders;
} xccl_hier_schedule_pairs_t;

typedef struct xccl_hier_bcast_spec {
    int                        use_sm_fanout_get; /**< Build the bcast schedule with FANOUT_GET innode
                                                      step instead of default BCAST */
    xccl_hier_schedule_pairs_t pairs;
    xccl_mem_h                 socket_memh; /**< mem descriptor of the socket group root to perform FANOUT_GET.
                                               Only used if "use_sm_fanout_get" is set. */
    xccl_mem_h                 socket_leaders_memh; /**< mem desc of the socket_leaders group root */
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

/* Makes a fragmented schedule of in_sched. The in_sched is stored as part of the created frag_sched
   and is cleaned_up automatically when frag_sched is finalized.
   @param [in] in_sched - input collective schedule
   @param [out] frag_sched - created fragmented schedule
   @param [in] binfo - original buffer information
   @param [in] frag_thresh - fragmentation threshold, or fragment size.
                             The size in bytes of the single fragment.
   @param [in] ordered - does the fragmented schedule should keep per-frag ordering
   @param [in] pipeline_depth - number of fragments outstanding at a time */
xccl_status_t make_fragmented_schedule(coll_schedule_t *in_sched, coll_schedule_t **frag_sched,
                                       xccl_coll_buffer_info_t binfo,
                                       size_t frag_thresh, int ordered, int pipeline_depth);

static inline
xccl_status_t coll_schedule_progress(coll_schedule_t *schedule) {
    return schedule->progress(schedule);
}
#endif
