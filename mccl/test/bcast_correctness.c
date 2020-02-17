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
    int root_base = argc > 1 ? atoi(argv[1]) : -1;

    const int max_count = 1 << 20;
    int *buf, *buf_mpi;
    int global_check;
    int iter;
    buf = (int*)malloc(max_count*sizeof(int));
    buf_mpi = (int*)malloc(max_count*sizeof(int));
    for (iter=0; iter<size; iter++) {
        for (int  count = max_count; count <=max_count; count *=2) {
            int root = root_base == -1 ? iter : root_base;
            if (rank == root) {
                for (int i=0; i<count; i++) {
                    buf[i] = (float)(rank+1);
                    buf_mpi[i] = (float)(rank+1);
                }
            } else {
                memset(buf, 0, count*sizeof(int));
            }

            mccl_bcast(buf, count, TCCL_DT_INT32, root, mccl_comm_world);
            MPI_Bcast(buf_mpi, count, MPI_INT, root, MPI_COMM_WORLD);
            int check = 0;
            for (int i=0; i<count ; i++) {
                if (buf[i] != buf_mpi[i]) {
                    fprintf(stderr, "Buf check error: pos %d, value %d, expected %d, count %d\n",
                            i, buf[i], buf_mpi[i], count);
                    check = 1;
                    break;
                }
            }

            MPI_Allreduce(&check, &global_check, 1 , MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (0 != global_check) {
                iter=size;
                break;
            }
        }
    }
    if (0 == rank) {
        printf("Result: %s\n", (0 == global_check) ? "SUCCESS" : "FAILURE");
    }
    free(buf);
    free(buf_mpi);
    mccl_test_fini();
    return 0;
}
