/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_mpi.h"
#include "test_utils.h"

int main (int argc, char **argv) {
    int rank, size, i, r, count,
        status = 0, status_global;
    int *sbuf, *rbuf, *sbuf_mpi, *rbuf_mpi;
    xccl_coll_req_h request;
    test_mem_type_t mtype;
    int not_equal;

    mtype = argc > 2 ? atoi(argv[2]) : TEST_MEM_TYPE_HOST;
    XCCL_CHECK(test_xccl_set_device(mtype));
    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLGATHER, XCCL_THREAD_MODE_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    count = argc > 1 ? atoi(argv[1]) : 32;
    count = size*((count + size - 1)/size);
    if (rank == 0) {
        test_print_header(XCCL_ALLGATHER, mtype, count, count);
    }

    XCCL_CHECK(test_xccl_mem_alloc((void**)&sbuf, count*sizeof(int), mtype));
    XCCL_CHECK(test_xccl_mem_alloc((void**)&rbuf, count*sizeof(int), mtype));
    sbuf_mpi = (int*)malloc(count*sizeof(int));
    rbuf_mpi = (int*)malloc(count*sizeof(int));

    for (i=0; i<count; i++) {
        sbuf_mpi[i] = rank + 1;
        rbuf_mpi[i] = 0;
    }
    XCCL_CHECK(test_xccl_memcpy(sbuf, sbuf_mpi, count*sizeof(int),
                                (mtype == TEST_MEM_TYPE_HOST) ? TEST_MEMCPY_H2H:
                                                                TEST_MEMCPY_H2D));
    XCCL_CHECK(test_xccl_memcpy(rbuf, rbuf_mpi, count*sizeof(int),
                                (mtype == TEST_MEM_TYPE_HOST) ? TEST_MEMCPY_H2H:
                                                                TEST_MEMCPY_H2D));

    xccl_coll_op_args_t coll = {
        .field_mask = 0,
        .coll_type = XCCL_ALLGATHER,
        .buffer_info = {
            .src_buffer = sbuf,
            .src_mtype  = UCS_MEMORY_TYPE_UNKNOWN,
            .dst_buffer = rbuf,
            .dst_mtype  = UCS_MEMORY_TYPE_UNKNOWN,
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

    MPI_Allgather(sbuf_mpi, count/size, MPI_INT, rbuf_mpi, count/size, MPI_INT, MPI_COMM_WORLD);
    XCCL_CHECK(test_xccl_memcmp(rbuf, mtype,
                                rbuf_mpi, TEST_MEM_TYPE_HOST,
                                count*sizeof(int),
                                &not_equal));
    if (not_equal) {
        fprintf(stderr, "RST CHECK FAILURE at rank %d, count %d\n",
                rank, count);
        status = 1;
    }
    MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }
    free(sbuf_mpi);
    free(rbuf_mpi);
    XCCL_CHECK(test_xccl_mem_free(sbuf, mtype));
    XCCL_CHECK(test_xccl_mem_free(rbuf, mtype));
    xccl_mpi_test_finalize();
    return 0;
}
