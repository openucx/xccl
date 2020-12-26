/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#define _BSD_SOURCE
#include "test_mpi.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static inline void
do_barrier(xccl_team_h team) {
    xccl_coll_req_h request;
    xccl_coll_op_args_t coll = {
        .field_mask = 0,
        .coll_type = XCCL_BARRIER,
        .alg.set_by_user = 0,
        .tag  = 123, //todo
    };
    XCCL_CHECK(xccl_collective_init(&coll, &request, team));
    XCCL_CHECK(xccl_collective_post(request));
    while (XCCL_OK != xccl_collective_test(request)) {
            xccl_context_progress(team_ctx);
        }
    XCCL_CHECK(xccl_collective_finalize(request));
}

int main (int argc, char **argv) {
    int rank, size, i, sleep_us;

    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_BARRIER, XCCL_THREAD_MODE_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));
    for (i=0; i<size; i++) {
        sleep_us = rand() % 1000;
        usleep(sleep_us);
        if (i == rank) {
            printf("Rank %d checks in\n", rank);
            fflush(stdout);
            usleep(100);
        }
        do_barrier(xccl_world_team);
    }

    xccl_mpi_test_finalize();
    return 0;
}
