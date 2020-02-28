#include "test_mpi.h"

xccl_team_h xccl_world_team;
static xccl_context_h team_ctx;

static int oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                         int my_rank, xccl_ep_range_t range,
                         void *coll_context) {
    MPI_Comm comm = (MPI_Comm)coll_context;
    if (XCCL_EP_RANGE_UNDEFINED == range.type) {
        MPI_Allgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm);
    } else {
        int root = xccl_range_to_rank(range, 0);
        if (my_rank == root) {
            int i;
            memcpy(rbuf, sbuf, msglen);
            for (i=1; i<range.ep_num; i++) {
                MPI_Recv((void*)((char*)rbuf + msglen*i), msglen,
                         MPI_BYTE, xccl_range_to_rank(range, i), 123,
                         comm, MPI_STATUS_IGNORE);
            }
            for (i=1; i<range.ep_num; i++) {
                MPI_Send(rbuf, msglen*range.ep_num, MPI_BYTE,
                         xccl_range_to_rank(range, i), 123, comm);
            }
        } else {
            MPI_Send(sbuf, msglen, MPI_BYTE, root, 123, comm);
            MPI_Recv(rbuf, msglen*range.ep_num, MPI_BYTE, root,
                     123, comm, MPI_STATUS_IGNORE);
        }
    }
    return 0;
}

int xccl_mpi_test_init(int argc, char **argv) {
    char *var;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Init xccl library */
    var = getenv("XCCL_TEST_TLS");
    xccl_params_t params = {
        .field_mask = XCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = XCCL_USAGE_SW_COLLECTIVES |
        XCCL_USAGE_HW_COLLECTIVES,
    };

    /* Init xccl context for a specified XCCL_TEST_TLS */
    xccl_config_t config = {
        .ctx_config = {
            .field_mask = XCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE |
            XCCL_CONTEXT_CONFIG_FIELD_OOB |
            XCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
            .thread_mode     = XCCL_LIB_THREAD_SINGLE,
            .completion_type = XCCL_TEAM_COMPLETION_BLOCKING,
            .oob = {
                .allgather    = oob_allgather,
                .coll_context = (void*)MPI_COMM_WORLD,
                .rank         = rank,
                .size         = size
            },
        },
        .tls = var, //NULL means auto
    };
    XCCL_CHECK(xccl_init(&params, &config, &team_ctx));
#if 0
    //TODO need to discuss where this should go
    xccl_team_lib_attr_t team_lib_attr;
    team_lib_attr.field_mask = XCCL_ATTR_FIELD_CONTEXT_CREATE_MODE;
    xccl_team_lib_query(lib, &team_lib_attr);
    if (team_lib_attr.context_create_mode == XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL) {
        xccl_oob_collectives_t oob_ctx = {
            .allgather  = oob_allgather,
            .coll_context = (void*)MPI_COMM_WORLD,
            .rank = rank,
            .size = size
        };

        team_ctx_config.oob = oob_ctx;
    }
#endif    

    /* Create XCCL TEAM for comm world */
    xccl_team_config_t team_config = {
        .range     = {
            .type           = XCCL_EP_RANGE_STRIDED,
            .strided.start  = 0,
            .strided.stride = 1
        }
    };

    xccl_oob_collectives_t oob = {
        .allgather  = oob_allgather,
        .coll_context = (void*)MPI_COMM_WORLD,
        .rank = rank,
        .size = size
    };

    XCCL_CHECK(xccl_team_create_post(team_ctx, &team_config, oob, &xccl_world_team));
    return 0;
}

int xccl_mpi_test_finalize(void) {
    XCCL_CHECK(xccl_team_destroy(xccl_world_team));
    XCCL_CHECK(xccl_cleanup(team_ctx));
    MPI_Finalize();
    return 0;
}
