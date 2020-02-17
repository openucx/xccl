/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <mpi.h>
#include "api/mccl.h"
#include <assert.h>
#include <string.h>

int oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                   int my_rank, int *ranks, int nranks,  void *oob_coll_ctx) {
    MPI_Comm comm = (MPI_Comm)oob_coll_ctx;
    if (ranks == NULL) {
        MPI_Allgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm);
    } else {
        if (my_rank == ranks[0]) {
            int i;
            memcpy(rbuf, sbuf, msglen);
            for (i=1; i<nranks; i++) {
                MPI_Recv((void*)((char*)rbuf + msglen*i), msglen, MPI_BYTE,
                         ranks[i], 123, comm, MPI_STATUS_IGNORE);
            }
            for (i=1; i<nranks; i++) {
                MPI_Send(rbuf, msglen*nranks, MPI_BYTE, ranks[i], 123, comm);
            }

        } else {
            MPI_Send(sbuf, msglen, MPI_BYTE, ranks[0], 123, comm);
            MPI_Recv(rbuf, msglen*nranks, MPI_BYTE, ranks[0], 123, comm, MPI_STATUS_IGNORE);
        }
    }
    return 0;
}

static mccl_context_h mccl_test_context;
mccl_comm_h mccl_comm_world;

int mccl_test_init(int argc, char **argv, uint64_t caps) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    mccl_config_t config = {
        .flags = 0,
        .world_size = size,
    };

    mccl_init_context(&config, &mccl_test_context);
    mccl_comm_config_t comm_config = {
        .allgather = oob_allgather,
        .oob_coll_ctx = (void*)MPI_COMM_WORLD,
        .mccl_ctx = mccl_test_context,
        .is_world = 1,
        .world_rank = rank,
        .comm_size  = size,
        .comm_rank  = rank,
        .caps.tagged_colls = 0,
    };

    mccl_comm_create(&comm_config, &mccl_comm_world);
    return MCCL_SUCCESS;
}

int mccl_test_fini(void) {
    mccl_comm_free(mccl_comm_world);
    mccl_finalize(mccl_test_context);
    MPI_Finalize();
    return MCCL_SUCCESS;
}
