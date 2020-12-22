/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

int main (int argc, char **argv) {
    size_t msg_size;
    xccl_coll_req_h request;
    int rank, size, i, status = 0, status_global, count, r;
    int *sbuf, *rbuf, *rbuf_mpi;

    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLREDUCE, XCCL_THREAD_MODE_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    count     = argc > 1 ? atoi(argv[1]) : 32;
    msg_size  = count*sizeof(int);
    sbuf      = (int*)malloc(msg_size);
    rbuf      = (int*)malloc(msg_size);
    rbuf_mpi  = (int*)malloc(msg_size);

    for (r=0; r<size; r++) {
        for (i=0; i<count; i++) {
            rbuf[i]     = 0;
            rbuf_mpi[i] = 0;
            sbuf[i]     = rank+1+12345 + i + r;
        }

        xccl_coll_op_args_t coll = {
            .field_mask = 0,
            .coll_type   = XCCL_REDUCE,
            .root        = r,
            .buffer_info = {
                .src_buffer = sbuf,
                .dst_buffer = rbuf,
                .len        = msg_size,
            },
            .reduce_info = {
                .dt         = XCCL_DT_INT32,
                .op         = XCCL_OP_SUM,
                .count      = count,
            },
            .alg.set_by_user = 0,
            .tag             = 123, //todo
        };

        XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
        XCCL_CHECK(xccl_collective_post(request));
        XCCL_CHECK(xccl_collective_wait(request));
        XCCL_CHECK(xccl_collective_finalize(request));

        MPI_Reduce(sbuf, rbuf_mpi, count, MPI_INT, MPI_SUM, r, MPI_COMM_WORLD);

        if (0 != memcmp(rbuf, rbuf_mpi, msg_size)) {
            fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
            status = 1;
        }

        MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (status_global != 0) {
            break;
        }
    }

    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }

    free(sbuf);
    free(rbuf);
    free(rbuf_mpi);
    xccl_mpi_test_finalize();
    return 0;
}
