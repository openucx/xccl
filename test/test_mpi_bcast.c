/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

int main (int argc, char **argv) {
    const int count =32;
    int rank, size, i, r, status = 0, status_global;
    int buf[count], buf_mpi[count];
    tccl_coll_req_h request;    
    tccl_mpi_test_init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (r=0; r<size; r++) {
        if (rank != r) {
            memset(buf, 0, sizeof(buf));
            memset(buf_mpi, 0, sizeof(buf_mpi));
        } else {
            for (i=0; i<count; i++) {
                buf[i] = buf_mpi[i] = rank+1+12345 + i;
            }
        }

        tccl_coll_op_args_t coll = {
            .coll_type = TCCL_BCAST,
            .root = r,
            .buffer_info = {
                .src_buffer = buf,
                .dst_buffer = buf,
                .len        = count*sizeof(int),
            },
            .alg.set_by_user = 1,
            .alg.id          = 1,
            .tag  = 123, //todo
        };

        tccl_collective_init(&coll, &request, tccl_world_team);
        tccl_collective_post(request);
        tccl_collective_wait(request);
        tccl_collective_finalize(request);

        MPI_Bcast(buf_mpi, count, MPI_INT, r, MPI_COMM_WORLD);
        if (0 != memcmp(buf, buf_mpi, count*sizeof(int))) {
            fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
            status = 1;
        }
        MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (0 != status_global) {
            break;
        }
    }

    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }

    tccl_mpi_test_finalize();
    return 0;
}
