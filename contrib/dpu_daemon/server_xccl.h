/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TEST_MPI_H
#define TEST_MPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <api/xccl.h>

#define DPU_XCCL_TLS "DPU_XCCL_TLS"

#define STR(x) # x
#define XCCL_CHECK(_call) if (XCCL_OK != (_call)) {              \
        fprintf(stderr, "*** XCCL TEST FAIL: %s\n", STR(_call)); \
        MPI_Abort(MPI_COMM_WORLD, -1);                           \
    }

typedef struct {
    xccl_team_h xccl_world_team;
    xccl_lib_h     lib;
    int rank, size;
    uint64_t tls;
} dpu_xccl_global_t;

typedef struct {
    dpu_xccl_global_t *g;
    xccl_context_h ctx;
    xccl_team_h team;
} dpu_xccl_comm_t;

int dpu_xccl_init(int argc, char **argv, dpu_xccl_global_t *g);
int dpu_xccl_alloc_team(dpu_xccl_global_t *g, dpu_xccl_comm_t *team);
int dpu_xccl_free_team(dpu_xccl_global_t *g, dpu_xccl_comm_t *ctx);
void dpu_xccl_finalize(dpu_xccl_global_t *g);
void dpu_xccl_progress(dpu_xccl_comm_t *team);

#endif
