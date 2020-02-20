#ifndef TEST_MPI_H
#define TEST_MPI_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <api/tccl.h>
#define STR(x) # x
#define TCCL_CHECK(_call) if (TCCL_OK != (_call)) {\
        fprintf(stderr, "fail: %s\n", STR(_call)); \
        exit(-1);                                  \
    }

extern tccl_team_h tccl_world_team;
int tccl_mpi_test_init(int argc, char **argv);
int tccl_mpi_test_finalize(void);

#endif
