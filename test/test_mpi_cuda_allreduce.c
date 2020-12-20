/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <cuda_runtime.h>
#include "test_mpi.h"

int main (int argc, char **argv) {
    const int count = 32;
    const int msg_size = count * sizeof(int);
    xccl_coll_req_h request;
    int rank, size, i, status = 0, status_global;
    int *sbuf_host, *sbuf_cuda, *rbuf_cuda, *rbuf_host, *rbuf_mpi;
    char *local_rank;
    cudaStream_t stream;

    local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank) {
        cudaSetDevice(atoi(local_rank));
    }
    cudaStreamCreate(&stream);

    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLREDUCE, XCCL_THREAD_MODE_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sbuf_host = (int*)malloc(msg_size);
    rbuf_mpi  = (int*)malloc(msg_size);
    rbuf_host = (int*)malloc(msg_size);
    cudaMalloc((void**)&sbuf_cuda, msg_size);
    cudaMalloc((void**)&rbuf_cuda, msg_size);

    for (i=0; i<count; i++) {
        rbuf_host[i] = 0;
        rbuf_mpi[i]  = 0;
        sbuf_host[i] = rank+1+12345 + i;
    }

    cudaMemcpyAsync(sbuf_cuda, sbuf_host, msg_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(rbuf_cuda, rbuf_host, msg_size, cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    xccl_coll_op_args_t coll = {
        .field_mask = 0,
        .coll_type   = XCCL_ALLREDUCE,
        .buffer_info = {
            .src_buffer = sbuf_cuda,
            .dst_buffer = rbuf_cuda,
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
    while (XCCL_OK != xccl_collective_test(request)) {
        xccl_context_progress(team_ctx);
    }
    XCCL_CHECK(xccl_collective_finalize(request));

    cudaMemcpyAsync(rbuf_host, rbuf_cuda, msg_size, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    MPI_Allreduce(sbuf_host, rbuf_mpi, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (0 != memcmp(rbuf_host, rbuf_mpi, msg_size)) {
        fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
        status = 1;
    }

    MPI_Reduce(&status, &status_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }

    cudaStreamDestroy(stream);
    cudaFree(sbuf_cuda);
    cudaFree(rbuf_cuda);
    free(sbuf_host);
    free(rbuf_host);
    free(rbuf_mpi);

    xccl_mpi_test_finalize();
    return 0;
}
