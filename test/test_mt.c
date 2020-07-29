/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "test_mpi.h"
#include <pthread.h>

void* do_allreduce(void *arg) {
    const int count = 32;
    xccl_coll_req_h request;
    int rank, size, i, status = 0, status_global, j;
    xccl_team_h team = (xccl_team_h)arg;
    int sbuf[count], rbuf[count];
    int iters = 10000;
    int check = 0;
    char *var = getenv("XCCL_TEST_ITERS");
    if (var) iters = atoi(var);
    var = getenv("XCCL_TEST_CHECK");
    if (var) check = atoi(var);
    
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

    for (i=0; i<iters; i++) {
        memset(rbuf, 0, sizeof(rbuf));
        XCCL_CHECK(xccl_collective_init(&coll, &request, team));
        XCCL_CHECK(xccl_collective_post(request));
        /* XCCL_CHECK(xccl_collective_wait(request)); */
        while (XCCL_OK != xccl_collective_test(request)) {
#ifdef CENTRAL_PROGRESS            
            xccl_context_progress(team_ctx);
#endif            
        }
        XCCL_CHECK(xccl_collective_finalize(request));
        if (check) {
            for (j=0; j<count; j++) {
                if (rbuf[j] != (size*(size+1)/2 + size*(j+12345))) {
                    fprintf(stderr, "DATA CHECK ERROR\n");
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
            }
        }
    }

    return NULL;
}


int main (int argc, char **argv) {
    char *var;
    int nthreads = 2, i;
    XCCL_CHECK(xccl_mpi_test_init(argc, argv, XCCL_COLL_CAP_ALLREDUCE));
    var = getenv("XCCL_TEST_NTHREADS");
    if (var) {
        nthreads = atoi(var);
    }
    xccl_team_h *teams;
    teams = malloc(nthreads*sizeof(*teams));
    for (i=0; i<nthreads; i++) {
        xccl_mpi_create_comm(MPI_COMM_WORLD, &teams[i]);
    }
    double t1 = MPI_Wtime();
    pthread_t *threads = malloc(nthreads*sizeof(*threads));
    for (i=0; i<nthreads; i++) {
        pthread_create(&threads[i], NULL, do_allreduce, teams[i]);
    }
    void *retval;
    for (i=0; i<nthreads; i++) {
        pthread_join(threads[i], &retval);
    }
    double elapsed = MPI_Wtime() - t1;
    for (i=0; i<nthreads; i++) {
        xccl_team_destroy(teams[i]);
    }
    double avg;
    MPI_Reduce(&elapsed, &avg, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (0 == rank) {
        printf("Elapsed %g us\n", i, avg*1e6/size);
    }
    xccl_mpi_test_finalize();
    return 0;
}
