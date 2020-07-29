/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

int main (int argc, char **argv) {
    const int count = 32;
    xccl_coll_req_h request;
    int rank, size, i, status = 0, status_global;
    int sbuf[count], rbuf[count], rbuf_mpi[count];

    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLREDUCE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (i=0; i<count; i++) {
        rbuf[i] = 0;
        sbuf[i] = rank+1+12345 + i;
    }

    xccl_coll_op_args_t coll = {
        .coll_type = XCCL_ALLREDUCE,
        .buffer_info = {
            .src_buffer = sbuf,
            .dst_buffer = rbuf,
            .len        = count*sizeof(int),
        },
        .reduce_info = {
            .dt = XCCL_DT_INT32,
            .op = XCCL_OP_SUM,
            .count = count,
        },
        .alg.set_by_user = 0,
        .tag  = 123, //todo
    };

    XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
    XCCL_CHECK(xccl_collective_post(request));
    /* XCCL_CHECK(xccl_collective_wait(request)); */
    while (XCCL_OK != xccl_collective_test(request)) {
        xccl_context_progress(team_ctx);
    }
    XCCL_CHECK(xccl_collective_finalize(request));

    MPI_Allreduce(sbuf, rbuf_mpi, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (0 != memcmp(rbuf, rbuf_mpi, count*sizeof(int))) {
        fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
        status = 1;
    }

    MPI_Reduce(&status, &status_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }

    xccl_mpi_test_finalize();
    return 0;
}
