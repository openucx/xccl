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
    int *buf, *buf_mpi;
    xccl_coll_req_h request;
    test_mem_type_t mtype;
    int not_equal;

    count = argc > 1 ? atoi(argv[1]) : 32;
    mtype = argc > 2 ? atoi(argv[2]) : TEST_MEM_TYPE_HOST;

    XCCL_CHECK(test_xccl_set_device(mtype));
    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_BCAST, XCCL_THREAD_MODE_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (rank == 0) {
        test_print_header(XCCL_BCAST, mtype, count, count);
    }

    XCCL_CHECK(test_xccl_mem_alloc((void**)&buf, count*sizeof(int),
                                   mtype));
    XCCL_CHECK(test_xccl_mem_alloc((void**)&buf_mpi, count*sizeof(int),
                                   TEST_MEM_TYPE_HOST));
    for (r=0; r<size; r++) {
        if (rank != r) {
            XCCL_CHECK(test_xccl_memset(buf, 0, count*sizeof(int),
                                        mtype))
            XCCL_CHECK(test_xccl_memset(buf_mpi, 0, count*sizeof(int),
                                        TEST_MEM_TYPE_HOST))
        } else {
            for (i=0; i<count; i++) {
                buf_mpi[i] = rank+1+12345 + i;
            }
            XCCL_CHECK(test_xccl_memcpy(buf, buf_mpi, count*sizeof(int),
                                        (mtype == TEST_MEM_TYPE_HOST) ? TEST_MEMCPY_H2H:
                                        TEST_MEMCPY_H2D));
        }

        xccl_coll_op_args_t coll = {
            .coll_type = XCCL_BCAST,
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

        XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
        XCCL_CHECK(xccl_collective_post(request));
        while (XCCL_OK != xccl_collective_test(request)) {
                xccl_context_progress(team_ctx);
            }
        XCCL_CHECK(xccl_collective_finalize(request));

        MPI_Bcast(buf_mpi, count, MPI_INT, r, MPI_COMM_WORLD);
        XCCL_CHECK(test_xccl_memcmp(buf, mtype,
                                    buf_mpi, TEST_MEM_TYPE_HOST,
                                    count*sizeof(int),
                                    &not_equal));

        if (not_equal) {
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
    XCCL_CHECK(test_xccl_mem_free(buf, mtype));
    XCCL_CHECK(test_xccl_mem_free(buf_mpi, TEST_MEM_TYPE_HOST));
    xccl_mpi_test_finalize();
    return 0;
}
