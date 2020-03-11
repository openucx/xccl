/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#define _BSD_SOURCE
#include "test_mpi.h"
#include <unistd.h>

int main (int argc, char **argv) {
    int rank, size, i, r, count,
        status = 0, status_global;
    int *buf, *buf_mpi;
    xccl_coll_req_h request;
    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_FANOUT_GET));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    count = argc > 1 ? atoi(argv[1]) : 32;
    buf = (int*)malloc(count*sizeof(int));
    buf_mpi = (int*)malloc(count*sizeof(int));

    const int max_iters = 5;
    int iters;
    for (r = 0; r<size; r++) {
        xccl_mem_h memh;
        xccl_mem_map_params_t params = {
            .field_mask = XCCL_MEM_MAP_PARAM_FIELD_ADDRESS |
            XCCL_MEM_MAP_PARAM_FIELD_LENGTH |
            XCCL_MEM_MAP_PARAM_FIELD_ROOT,
            .address = buf,
            .length  = count*sizeof(int),
            .root = r,
        };
        XCCL_CHECK(xccl_global_mem_map_start(xccl_world_team, params, &memh));
        while (XCCL_OK != xccl_global_mem_map_test(memh)) {;}
        for (iters=0; iters < max_iters; iters++) {
            if (rank != r) {
                memset(buf, 0, sizeof(int)*count);
                memset(buf_mpi, 0, sizeof(int)*count);
            } else {
                for (i=0; i<count; i++) {
                    buf[i] = buf_mpi[i] = rank+1+12345 + i + 11*iters;
                }
                usleep(10000);
            }

            size_t half = count*sizeof(int)/2;
            xccl_coll_op_args_t coll = {
                .coll_type = XCCL_FANOUT_GET,
                .root = r,
                .get_info = {
                    .memh       = memh,
                    .offset     = half,
                    .local_buffer = (void*)((ptrdiff_t)&buf[0] + half),
                    .len        = half,
                },
            };
            XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
            XCCL_CHECK(xccl_collective_post(request));
            XCCL_CHECK(xccl_collective_wait(request));
            XCCL_CHECK(xccl_collective_finalize(request));

            coll.get_info.offset = 0;
            coll.get_info.local_buffer = buf;
            XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
            XCCL_CHECK(xccl_collective_post(request));
            XCCL_CHECK(xccl_collective_wait(request));
            XCCL_CHECK(xccl_collective_finalize(request));

            MPI_Bcast(buf_mpi, count, MPI_INT, r, MPI_COMM_WORLD);
            if (0 != memcmp(buf, buf_mpi, count*sizeof(int))) {
                fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
                status = 1;
            }
            MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (0 != status_global) {
                r = size;
                break;
            }
        }
        XCCL_CHECK(xccl_global_mem_unmap(memh));
    }

    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }
    free(buf);
    free(buf_mpi);
    XCCL_CHECK(xccl_mpi_test_finalize());
    return 0;
}
