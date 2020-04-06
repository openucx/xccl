/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <cuda_runtime.h>
#include "test_mpi.h"

int main (int argc, char **argv) {
    int rank, size, i, r, count,
        status = 0, status_global;
    int *buf, *buf_mpi, *buf_cuda_src, *buf_cuda_dst;
    int msg_size;
    cudaStream_t stream;
    char *local_rank;

    xccl_coll_req_h request;    
    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_BCAST));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank) {
        cudaSetDevice(atoi(local_rank));
    }
    cudaStreamCreate(&stream);

    count = argc > 1 ? atoi(argv[1]) : 32;
    msg_size = count * sizeof(int);
    buf = (int*)malloc(msg_size);
    buf_mpi = (int*)malloc(msg_size);
    cudaMalloc((void**)&buf_cuda_src, msg_size);
    cudaMalloc((void**)&buf_cuda_dst, msg_size);

    for (r=0; r<size; r++) {
        if (rank != r) {
            memset(buf, 0, msg_size);
            memset(buf_mpi, 0, msg_size);
        } else {
            for (i=0; i<count; i++) {
                buf[i] = buf_mpi[i] = rank+1+12345 + i;
            }
        }
        cudaMemcpyAsync(buf_cuda_src, buf, msg_size, cudaMemcpyHostToDevice, stream);
        cudaMemsetAsync(buf_cuda_dst, 0, msg_size, stream);
        cudaStreamSynchronize(stream);

        xccl_coll_op_args_t coll = {
            .coll_type = XCCL_BCAST,
            .root = r,
            .buffer_info = {
                .src_buffer = buf_cuda_src,
                .dst_buffer = buf_cuda_dst,
                .len        = msg_size,
            },
            .alg.set_by_user = 1,
            .alg.id          = 1,
            .tag  = 123, //todo
        };

        XCCL_CHECK(xccl_collective_init(&coll, &request, xccl_world_team));
        XCCL_CHECK(xccl_collective_post(request));
        XCCL_CHECK(xccl_collective_wait(request));
        XCCL_CHECK(xccl_collective_finalize(request));

        cudaMemcpyAsync(buf, buf_cuda_dst, msg_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        MPI_Bcast(buf_mpi, count, MPI_INT, r, MPI_COMM_WORLD);
        if (0 != memcmp(buf, buf_mpi, msg_size)) {
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
    cudaFree(buf_cuda_src);
    cudaFree(buf_cuda_dst);
    free(buf);
    free(buf_mpi);
    XCCL_CHECK(xccl_mpi_test_finalize());
    return 0;
}
