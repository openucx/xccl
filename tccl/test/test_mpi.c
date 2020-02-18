/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "api/tccl.h"

#define STR(x) # x
#define TCCL_CHECK(_call) if (TCCL_OK != (_call)) {\
        fprintf(stderr, "fail: %s\n", STR(_call)); \
        exit(-1);                                  \
    }

static int oob_allgather(void *sbuf, void *rbuf, size_t len, void *coll_context) {
    MPI_Comm comm = (MPI_Comm)coll_context;
    MPI_Allgather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, comm);
    return 0;
}
int main (int argc, char **argv) {
    int rank, size;
    char *var;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    var = getenv("TCCL_TEST_TEAM");
    tccl_lib_config_t lib_config = {
        .field_mask = TCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = TCCL_USAGE_SW_COLLECTIVES |
                      TCCL_USAGE_HW_COLLECTIVES,
    };
    tccl_lib_h lib;
    tccl_lib_init(lib_config, &lib);

    tccl_team_context_config_t team_ctx_config = {
        .field_mask = TCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME |
                      TCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE |
                      TCCL_CONTEXT_CONFIG_FIELD_OOB |
                      TCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
        .team_lib_name   = var ? var : "ucx",
        .thread_mode     = TCCL_LIB_THREAD_SINGLE,
        .completion_type = TCCL_TEAM_COMPLETION_BLOCKING,
        .oob = {
            .allgather    = oob_allgather,
            .coll_context = (void*)MPI_COMM_WORLD,
            .rank         = rank,
            .size         = size
        },
    };
#if 0
    //TODO
    tccl_team_lib_attr_t team_lib_attr;
    team_lib_attr.field_mask = TCCL_ATTR_FIELD_CONTEXT_CREATE_MODE;
    tccl_team_lib_query(lib, &team_lib_attr);
    if (team_lib_attr.context_create_mode == TCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL) {
        tccl_oob_collectives_t oob_ctx = {
            .allgather  = oob_allgather,
            .coll_context = (void*)MPI_COMM_WORLD,
            .rank = rank,
            .size = size
        };

        team_ctx_config.oob = oob_ctx;
    }
#endif    
    tccl_team_context_h team_ctx;
    TCCL_CHECK(tccl_create_context(lib, team_ctx_config, &team_ctx));
    {
        /* Create TEAM for comm world */
        tccl_team_config_t team_config = {
            .team_size = size,
            .team_rank = rank,
            .range     = {
                .type           = TCCL_EP_RANGE_STRIDED,
                .strided.start  = 0,
                .strided.stride = 1
            }
        };

        tccl_oob_collectives_t oob = {
            .allgather  = oob_allgather,
            .coll_context = (void*)MPI_COMM_WORLD,
            .rank = rank,
            .size = size
        };

        tccl_team_h world_team;
        tccl_team_create_post(team_ctx, &team_config, oob, &world_team);
        const int count =32;
        int sbuf[count], rbuf[count], rbuf_mpi[count];
        int i;
        for (i=0; i<count; i++) {
            rbuf[i] = 0;
            sbuf[i] = rank+1+12345 + i;
        }
        tccl_coll_req_h request;
        tccl_coll_op_args_t coll = {
            .coll_type = TCCL_ALLREDUCE,
            .buffer_info = {
                .src_buffer = sbuf,
                .dst_buffer = rbuf,
                .len        = count*sizeof(int),
            },
            .reduce_info = {
                .dt = TCCL_DT_INT32,
                .op = TCCL_OP_SUM,
                .count = count,
            },
            .alg.set_by_user = 0,
            .tag  = 123, //todo
        };
        tccl_collective_init(&coll, &request, world_team);
        tccl_collective_post(request);
        tccl_collective_wait(request);
        tccl_collective_finalize(request);
        tccl_team_destroy(world_team);

        MPI_Allreduce(sbuf, rbuf_mpi, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        int status = 0, status_global;
        if (0 != memcmp(rbuf, rbuf_mpi, count*sizeof(int))) {
            fprintf(stderr, "RST CHECK FAILURE at rank %d\n", rank);
            status = 1;
        }
        MPI_Reduce(&status, &status_global, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        if (0 == rank) {
            printf("Correctness check: %s\n", status_global == 0 ? "PASS" : "FAIL");
        }
    }
    tccl_destroy_team_context(team_ctx);
    MPI_Finalize();
}
