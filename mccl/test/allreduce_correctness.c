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

    const int max_count = 256;
    int global_check;
    for (int  count = 1; count <=max_count; count *=2) {
        float sbuf[count], rbuf[count], rbuf_mpi[count];
        for (int i=0; i<count; i++) {
            sbuf[i] = (float)(rank+1);
            rbuf[i] = rbuf_mpi[i] = 0;
        }
        mccl_allreduce(sbuf, rbuf, count, TCCL_DT_FLOAT32, TCCL_OP_SUM, mccl_comm_world);
        MPI_Allreduce(sbuf, rbuf_mpi, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        int check = 0;
        for (int i=0; i<count ; i++) {
            if (fabs(rbuf[i] -  rbuf_mpi[i])/fabs(rbuf_mpi[i]) > 0.01) {
                fprintf(stderr, "Buf check error: pos %d, value %g, expected %g, count %d\n",
                        i, rbuf[i], rbuf_mpi[i], count);
                check = 1;
                break;
            }
        }

        MPI_Allreduce(&check, &global_check, 1 , MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (0 != global_check) {
            break;
        }
    }
    if (0 == rank) {
        printf("Result: %s\n", (0 == global_check) ? "SUCCESS" : "FAILURE");
    }

    mccl_test_fini();
    return 0;
}
