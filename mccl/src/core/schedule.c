#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "schedule.h"
#include "mccl_team.h"

mccl_status_t build_allreduce_schedule_3lvl(mccl_comm_t *comm, coll_schedule_t **sched,
                                            coll_schedule_type_t sched_type, int count,
                                            tccl_dt_t dtype, tccl_op_t op, void *sbuf, void *rbuf,
                                            int socket_teamtype, int socket_leaders_teamtype,
                                            int node_leaders_teamtype) {
    int have_node_leaders_group = (comm->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group = (comm->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (comm->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);

    int node_leaders_group_exists = (comm->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists = (comm->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (comm->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    sbgp_type_t top_sbgp = node_leaders_group_exists ? SBGP_NODE_LEADERS :
        (socket_leaders_group_exists ? SBGP_SOCKET_LEADERS : SBGP_SOCKET);

    coll_schedule_sequential_t *schedule = (coll_schedule_sequential_t *)malloc(sizeof(*schedule));
    schedule->super.comm = comm;
    schedule->super.type = sched_type;
    int c = 0;
    tccl_coll_op_args_t coll = {
        .coll_type = TCCL_REDUCE,
        .buffer_info = {
            .src_buffer = sbuf,
            .dst_buffer = rbuf,
            .len        = count*tccl_dt_size(dtype),
        },
        .reduce_info = {
            .dt    = dtype,
            .op    = op,
            .count = count,
        },
        .root            = 0,
        .alg.set_by_user = 0,
        .tag             = 123, //todo
    };

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = TCCL_ALLREDUCE;
            schedule->super.args[c].tccl_coll = coll;
        } else {
            coll.coll_type = TCCL_REDUCE;
            schedule->super.args[c].tccl_coll = coll;
        }
        schedule->super.args[c].team = comm->teams[socket_teamtype];
        c++;
        coll.buffer_info.src_buffer = rbuf;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = TCCL_ALLREDUCE;
            schedule->super.args[c].tccl_coll = coll;
        } else {
            coll.coll_type = TCCL_REDUCE;
            schedule->super.args[c].tccl_coll = coll;
        }
        schedule->super.args[c].team = comm->teams[socket_leaders_teamtype];
        c++;
        coll.buffer_info.src_buffer = rbuf;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = TCCL_ALLREDUCE;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[node_leaders_teamtype];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = TCCL_BCAST;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[socket_leaders_teamtype];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = TCCL_BCAST;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[socket_teamtype];
        c++;
    }
    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    schedule->req = NULL;
    (*sched) = &schedule->super;
    return MCCL_SUCCESS;
}

mccl_status_t build_barrier_schedule_3lvl(mccl_comm_t *comm, coll_schedule_t **sched,
                                          coll_schedule_type_t sched_type,
                                          int socket_teamtype, int socket_leaders_teamtype,
                                          int node_leaders_teamtype) {
    int have_node_leaders_group = (comm->sbgps[SBGP_NODE_LEADERS].status == SBGP_ENABLED);
    int have_socket_group = (comm->sbgps[SBGP_SOCKET].status == SBGP_ENABLED);
    int have_socket_leaders_group = (comm->sbgps[SBGP_SOCKET_LEADERS].status == SBGP_ENABLED);

    int node_leaders_group_exists = (comm->sbgps[SBGP_NODE_LEADERS].status != SBGP_NOT_EXISTS);
    int socket_group_exists = (comm->sbgps[SBGP_SOCKET].status != SBGP_NOT_EXISTS);
    int socket_leaders_group_exists = (comm->sbgps[SBGP_SOCKET_LEADERS].status != SBGP_NOT_EXISTS);
    sbgp_type_t top_sbgp = node_leaders_group_exists ? SBGP_NODE_LEADERS :
        (socket_leaders_group_exists ? SBGP_SOCKET_LEADERS : SBGP_SOCKET);

    coll_schedule_sequential_t *schedule = (coll_schedule_sequential_t *)malloc(sizeof(*schedule));
    schedule->super.comm = comm;
    schedule->super.type = sched_type;
    int c = 0;
    tccl_coll_op_args_t coll = {
        .coll_type       = TCCL_BARRIER,
        .alg.set_by_user = 0,
        .tag             = 123, //todo
    };

    if (have_socket_group) {
        if (top_sbgp == SBGP_SOCKET) {
            coll.coll_type = TCCL_BARRIER;
            schedule->super.args[c].tccl_coll = coll;
        } else {
            coll.coll_type = TCCL_FANIN;
            schedule->super.args[c].tccl_coll = coll;
        }
        schedule->super.args[c].team = comm->teams[socket_teamtype];
        c++;
    }

    if (have_socket_leaders_group) {
        if (top_sbgp == SBGP_SOCKET_LEADERS) {
            coll.coll_type = TCCL_BARRIER;
            schedule->super.args[c].tccl_coll = coll;
        } else {
            coll.coll_type = TCCL_FANIN;
            schedule->super.args[c].tccl_coll = coll;
        }
        schedule->super.args[c].team = comm->teams[socket_leaders_teamtype];
        c++;
    }

    if (have_node_leaders_group) {
        assert(top_sbgp == SBGP_NODE_LEADERS);
        coll.coll_type = TCCL_BARRIER;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[node_leaders_teamtype];
        c++;
    }

    if (have_socket_leaders_group && top_sbgp != SBGP_SOCKET_LEADERS) {
        coll.coll_type = TCCL_FANOUT;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[socket_leaders_teamtype];
        c++;
    }

    if (have_socket_group  && top_sbgp != SBGP_SOCKET) {
        coll.coll_type = TCCL_FANOUT;
        schedule->super.args[c].tccl_coll = coll;
        schedule->super.args[c].team = comm->teams[socket_teamtype];
        c++;
    }
    schedule->super.n_colls = c;
    schedule->super.n_completed_colls = 0;
    schedule->req = NULL;
    (*sched) = &schedule->super;
    return MCCL_SUCCESS;
}

static inline mccl_status_t
coll_schedule_progress_sequential(coll_schedule_sequential_t *schedule) {
    int i;
    int curr_idx;
    mccl_coll_args_t *curr_op;
    int n_colls = schedule->super.n_colls;
    const int n_polls = 10;
    for (i=0; i<n_polls && n_colls != schedule->super.n_completed_colls; i++) {
        curr_idx = schedule->super.n_completed_colls;
        curr_op = &schedule->super.args[curr_idx];
        if (!schedule->req) {
            tccl_collective_init(&curr_op->tccl_coll, &schedule->req,
                                curr_op->team->tccl_team);
            tccl_collective_post(schedule->req);
        }
        if (TCCL_OK == tccl_collective_test(schedule->req)) {
            schedule->super.n_completed_colls++;
            tccl_collective_finalize(schedule->req);
            schedule->req = NULL;
            i = 0;
        }
    }
    return MCCL_SUCCESS;
}

static inline mccl_status_t
coll_schedule_progress_single_dep(coll_schedule_single_dep_t *schedule) {
    int i, p;
    int curr_idx;
    mccl_coll_args_t *curr_op;
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
            /*         curr_op->tccl_coll.root); */

            tccl_collective_init(&curr_op->tccl_coll, &schedule->reqs[curr_idx],
                                curr_op->team->tccl_team);
            tccl_collective_post(schedule->reqs[curr_idx]);
        }
        for (p=0; p<n_polls; p++) {
            if (TCCL_OK == tccl_collective_test(schedule->reqs[curr_idx])) {
                /* fprintf(stderr, "completed first dep, starting all others: %d\n", */
                /*         n_colls-1); */
                schedule->super.n_completed_colls++;
                tccl_collective_finalize(schedule->reqs[curr_idx]);
                schedule->reqs[curr_idx] = NULL;
                for (i=0; i<n_colls; i++) {
                    if (i != curr_idx) {
                        curr_op = &schedule->super.args[i];
                        assert(NULL == schedule->reqs[i]);
                        /* fprintf(stderr,"starting [%d]: %s, group_rank %d, l_root %d\n", i, */
                        /*         sbgp_type_str[curr_op->team->sbgp->type], */
                        /*         curr_op->team->sbgp->group_rank, */
                        /*         curr_op->tccl_coll.root); */

                        tccl_collective_init(&curr_op->tccl_coll, &schedule->reqs[i],
                                            curr_op->team->tccl_team);
                        tccl_collective_post(schedule->reqs[i]);
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
                    if (TCCL_OK == tccl_collective_test(schedule->reqs[i])) {
                        schedule->super.n_completed_colls++;
                        /* fprintf(stderr, "completed [%d], %s, n_completed %d\n", */
                        /*         i, sbgp_type_str[schedule->super.args[i].team->sbgp->type], */
                        /*         schedule->super.n_completed_colls); */
                        tccl_collective_finalize(schedule->reqs[i]);
                        schedule->reqs[i] = NULL;
                        p = 0;
                    }
                }
            }
        }
    }
    return MCCL_SUCCESS;
}

mccl_status_t coll_schedule_progress(coll_schedule_t *schedule) {
    switch(schedule->type) {
    case MCCL_COLL_SCHED_SEQ:
        return coll_schedule_progress_sequential((coll_schedule_sequential_t*)schedule);
    case MCCL_COLL_SCHED_SINGLE_DEP:
        return coll_schedule_progress_single_dep((coll_schedule_single_dep_t*)schedule);
    default:
        break;
    }
    return MCCL_ERROR;
}

mccl_status_t mccl_start(mccl_request_h req) {
    coll_schedule_t *schedule = (coll_schedule_t*)req;
    coll_schedule_progress(schedule);
    return MCCL_SUCCESS;
}

mccl_status_t mccl_test(mccl_request_h req) {
    coll_schedule_t *schedule = (coll_schedule_t*)req;
    coll_schedule_progress(schedule);
    return schedule->n_completed_colls == schedule->n_colls ?
        MCCL_SUCCESS : MCCL_IN_PROGRESS;
}

mccl_status_t mccl_wait(mccl_request_h req) {
    int ret = mccl_test(req);
    while (MCCL_SUCCESS != ret) {
        ret = mccl_test(req);
    }
    return MCCL_SUCCESS;
}

mccl_status_t mccl_request_free(mccl_request_h req) {
    coll_schedule_t *schedule = (coll_schedule_t*)req;
    free(schedule);//todo mpool
    return MCCL_SUCCESS;
}

