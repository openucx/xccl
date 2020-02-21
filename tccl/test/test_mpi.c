#include "test_mpi.h"

tccl_team_h tccl_world_team;

static tccl_lib_h lib;
static tccl_context_h team_ctx;

static int oob_allgather(void *sbuf, void *rbuf, size_t len, void *coll_context) {
    MPI_Comm comm = (MPI_Comm)coll_context;
    MPI_Allgather(sbuf, len, MPI_BYTE, rbuf, len, MPI_BYTE, comm);
    return 0;
}

int tccl_mpi_test_init(int argc, char **argv) {
    char *var;
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Init tccl library */
    var = getenv("TCCL_TEST_TEAM");
    tccl_lib_config_t lib_config = {
        .field_mask = TCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = TCCL_USAGE_SW_COLLECTIVES |
        TCCL_USAGE_HW_COLLECTIVES,
    };
    TCCL_CHECK(tccl_lib_init(lib_config, &lib));

    /* Init tccl context for a specified TCCL_TEST_TEAM */
    tccl_context_config_t team_ctx_config = {
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
    TCCL_CHECK(tccl_create_context(lib, team_ctx_config, &team_ctx));

#if 0
    //TODO need to discuss where this should go
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

    /* Create TCCL TEAM for comm world */
    tccl_team_config_t team_config = {
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

    TCCL_CHECK(tccl_team_create_post(team_ctx, &team_config, oob, &tccl_world_team));
    return 0;
}

int tccl_mpi_test_finalize(void) {
    TCCL_CHECK(tccl_team_destroy(tccl_world_team));
    TCCL_CHECK(tccl_destroy_context(team_ctx));
    TCCL_CHECK(tccl_lib_finalize(lib));
    MPI_Finalize();
    return 0;
}
