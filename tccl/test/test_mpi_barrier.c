/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

static inline void
do_barrier(tccl_team_h team) {
    tccl_coll_req_h request;
    tccl_coll_op_args_t coll = {
        .coll_type = TCCL_BARRIER,
        .alg.set_by_user = 0,
        .tag  = 123, //todo
    };
    tccl_collective_init(&coll, &request, team);
    tccl_collective_post(request);
    tccl_collective_wait(request);
    tccl_collective_finalize(request);
}

int main (int argc, char **argv) {
    int rank, size, i, sleep_us;

    tccl_mpi_test_init(argc, argv);
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
        do_barrier(tccl_world_team);
    }

    tccl_mpi_test_finalize();
}
