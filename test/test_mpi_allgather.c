/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

int main (int argc, char **argv) {
    int rank, size, i, r, count,
        status = 0, status_global;
    int *sbuf, *rbuf, *rbuf_mpi;
    xccl_coll_req_h request;    
    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLGATHER, XCCL_THREAD_MODE_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    count = argc > 1 ? atoi(argv[1]) : 32;
    count = size*((count + size - 1)/size);

    sbuf = (int*)malloc(count*sizeof(int));
    rbuf = (int*)malloc(count*sizeof(int));
    rbuf_mpi = (int*)malloc(count*sizeof(int));

    for (i=0; i<count; i++) {
        sbuf[i] = rank + 1;
        rbuf[i] = rbuf_mpi[i] = 0;                                                                                                                                                                                                                                                                                                  
    }                                                   

    xccl_coll_op_args_t coll = {
        .coll_type = XCCL_ALLGATHER,
        .buffer_info = {
            .src_buffer = sbuf,
            .dst_buffer = rbuf,
            .len        = count*sizeof(int),
        },
        .alg.set_by_user = 0,
        .tag  = 123, //todo
    };

    XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
    XCCL_CHECK(xccl_collective_post(request));
    while (XCCL_OK != xccl_collective_test(request)) {
        xccl_context_progress(team_ctx);
    }
    XCCL_CHECK(xccl_collective_finalize(request));

    MPI_Allgather(sbuf, count/size, MPI_INT, rbuf_mpi, count/size, MPI_INT, MPI_COMM_WORLD);
    if (0 != memcmp(rbuf, rbuf_mpi, count*sizeof(int))) {
        fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
        status = 1;
    }
    MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }
    free(sbuf);
    free(rbuf);
    free(rbuf_mpi);
    xccl_mpi_test_finalize();
    return 0;
}
