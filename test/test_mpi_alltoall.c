/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"

int run_test(void *sbuf, void *rbuf, void *rbuf_mpi, int count, int rank)
{
    xccl_coll_req_h request;
    MPI_Request     mpi_req;
    int             status, status_global, completed;
    int i = 0;

    status = 0;
    xccl_coll_op_args_t coll = {
        .coll_type = XCCL_ALLTOALL,
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

    if (sbuf != rbuf) {
        MPI_Ialltoall(sbuf, count, MPI_INT, rbuf_mpi, count, MPI_INT, MPI_COMM_WORLD, &mpi_req);
    } else {
        MPI_Ialltoall(MPI_IN_PLACE, count, MPI_INT, rbuf_mpi, count, MPI_INT, MPI_COMM_WORLD, &mpi_req);
    }
    completed = 0;
    while (!completed) {
        MPI_Test(&mpi_req, &completed, MPI_STATUS_IGNORE);
        xccl_mpi_test_progress();
    }

    if (0 != memcmp(rbuf, rbuf_mpi, count*sizeof(int))) {

        fprintf(stderr, "RST CHECK FAILURE at rank %d, count %d\n", rank, count);
        status = 1;
    }

    MPI_Allreduce(&status, &status_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return status_global;
}

int main (int argc, char **argv)
{
    const int iters = 5;
    size_t msglen_min, msglen_max;
    int count_max, count_min, count,
        rank, size, i, status_global;
    int *sbuf, *rbuf, *rbuf_mpi;
    msglen_min = argc > 1 ? atoi(argv[1]) : 4;
    msglen_max = argc > 2 ? atoi(argv[2]) : 1024;
    if (msglen_max < msglen_min) {
        fprintf(stderr, "Incorrect msglen settings\n");
        return -1;
    }
    count_max = (msglen_max + sizeof(int) - 1)/sizeof(int);
    count_min = (msglen_min + sizeof(int) - 1)/sizeof(int);

    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLTOALL));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    sbuf     = malloc(count_max*size*sizeof(int));
    rbuf     = malloc(count_max*size*sizeof(int));
    rbuf_mpi = malloc(count_max*size*sizeof(int));
    for (i=0; i<count_max*size; i++) {
        sbuf[i] = rank+1;
    }
/* regular alltoall */
    for (count = count_min; count <= count_max; count *= 2) {
        for (i=0; i<iters; i++) {
            memset(rbuf, 0, sizeof(count*size*sizeof(int)));
            status_global = run_test(sbuf, rbuf, rbuf_mpi, count, rank);
            if (status_global) {
                goto end;
            }
        }
        count *= 2;
    }

/* in-place alltoall */
    for (count = count_min; count <= count_max; count *= 2) {
        for (i=0; i<iters; i++) {
            memcpy(rbuf,     sbuf, count*size*sizeof(int));
            memcpy(rbuf_mpi, sbuf, count*size*sizeof(int));
            status_global = run_test(rbuf, rbuf, rbuf_mpi, count, rank);
            if (status_global) {
                goto end;
            }
        }
        count *= 2;
    }

end:
    if (0 == rank) {
        printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
    }

    free(sbuf);
    free(rbuf);
    free(rbuf_mpi);
    xccl_mpi_test_finalize();
    return 0;
}
