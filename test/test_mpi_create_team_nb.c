/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

int main (int argc, char **argv) {
    int rank, size;
    int sbuf, rbuf;

    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLREDUCE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm split_comm;
    xccl_team_h split_team;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &split_comm);
    int split_rank, split_size;
    MPI_Comm_rank(split_comm, &split_rank);
    MPI_Comm_size(split_comm, &split_size);

    if (split_rank % 2 == 0) {
        MPI_Sendrecv(&sbuf, 1, MPI_INT, (split_rank + 1) % split_size, 123,
                     &rbuf, 1, MPI_INT, (split_rank + split_size - 1) % split_size, 123,
                     split_comm, MPI_STATUS_IGNORE);
        xccl_mpi_create_comm_nb(split_comm, &split_team);
    } else {
        xccl_mpi_create_comm_nb(split_comm, &split_team);
        MPI_Sendrecv(&sbuf, 1, MPI_INT, (split_rank + 1) % split_size, 123,
                     &rbuf, 1, MPI_INT, (split_rank + split_size - 1) % split_size, 123,
                     split_comm, MPI_STATUS_IGNORE);
    }
    while (XCCL_INPROGRESS == xccl_team_create_test(split_team)) {;};
    XCCL_CHECK(xccl_team_destroy(split_team));
        MPI_Comm_free(&split_comm);
    if (0 == rank) {
        printf("Correctness check: %s\n", "PASS");
    }

    XCCL_CHECK(xccl_mpi_test_finalize());
    return 0;
}
