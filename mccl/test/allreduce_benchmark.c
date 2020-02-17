/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#define _BSD_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include "api/mccl.h"
#include "mccl_test.h"

int main (int argc, char **argv) {
    int rank, size;
    mccl_test_init(argc, argv, 0);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t msglen = (size_t)(argc > 1 ? atoi(argv[1]) : 4);
    int warmups = argc > 2 ? atoi(argv[2]) : 100;
    int iters = argc > 3 ? atoi(argv[3]) : 1000;
    int use_mpi = argc > 4 ? atoi(argv[4]) : 0;
    int count = msglen/sizeof(float);
    void *sbuf, *rbuf;
    sbuf = malloc(msglen);
    rbuf = malloc(msglen);

    double t = 0, t1;;
    for (int i=0; i<warmups+iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        if (use_mpi) {
            MPI_Allreduce(sbuf, rbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        } else {
            mccl_allreduce(sbuf, rbuf, count, TCCL_DT_FLOAT32, TCCL_OP_SUM, mccl_comm_world);
        }
        if (i>= warmups) {
            t += MPI_Wtime() - t1;
        }
    }

    t = t/iters;
    double t_min, t_max, t_av;
    MPI_Reduce(&t, &t_min, 1 , MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t, &t_max, 1 , MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t, &t_av, 1 , MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (0 == rank) {
        printf("msglen\tavg\tmin\tmax\n");
        printf("%zd\t%g\t%g\t%g\n", msglen, t_av*1e6/size, t_min*1e6, t_max*1e6);
    }

    free(sbuf);free(rbuf);
    mccl_test_fini();
    return 0;
}
