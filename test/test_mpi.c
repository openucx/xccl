#include "test_mpi.h"

xccl_team_h xccl_world_team;
static xccl_context_h team_ctx;

typedef struct xccl_test_oob_allgather_req {
    xccl_ep_range_t range;
    void *sbuf;
    void *rbuf;
    void *oob_coll_ctx;
    int my_rank;
    size_t msglen;
    int iter;
    MPI_Request reqs[2];
} xccl_test_oob_allgather_req_t;

static xccl_status_t oob_allgather_test(void *req)
{
    xccl_test_oob_allgather_req_t *oob_req =
        (xccl_test_oob_allgather_req_t*)req;
    int rank, size, sendto, recvfrom, recvdatafrom, senddatafrom, completed, probe;
    char *tmpsend = NULL, *tmprecv = NULL;
    size_t msglen = oob_req->msglen;
    const int probe_count = 1;
    MPI_Comm comm = (MPI_Comm)oob_req->oob_coll_ctx;

    if (oob_req->range.type == XCCL_EP_RANGE_UNDEFINED) {
        MPI_Comm_size(comm, &size);
        MPI_Comm_rank(comm, &rank);
    } else {
        size = oob_req->range.ep_num;
        rank = oob_req->my_rank;
    }
    if (oob_req->iter == 0) {
        tmprecv = (char*) oob_req->rbuf + (ptrdiff_t)rank * (ptrdiff_t)msglen;
        memcpy(tmprecv, oob_req->sbuf, msglen);
    }
    sendto = (rank + 1) % size;
    recvfrom  = (rank - 1 + size) % size;
    if (oob_req->range.type != XCCL_EP_RANGE_UNDEFINED) {
        sendto = xccl_range_to_rank(oob_req->range, sendto);
        recvfrom = xccl_range_to_rank(oob_req->range, recvfrom);
    }
    for (; oob_req->iter < size - 1; oob_req->iter++) {
        if (oob_req->iter > 0) {
            probe = 0;
            do {
                MPI_Testall(2, oob_req->reqs, &completed, MPI_STATUS_IGNORE);
                probe++;
            } while (!completed && probe < probe_count);
            if (!completed) {
                return XCCL_INPROGRESS;
            }
        }
        recvdatafrom = (rank - oob_req->iter - 1 + size) % size;
        senddatafrom = (rank - oob_req->iter + size) % size;
        tmprecv = (char*)oob_req->rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)msglen;
        tmpsend = (char*)oob_req->rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)msglen;
        MPI_Isend(tmpsend, msglen, MPI_BYTE, sendto, 2703,
                  comm, &oob_req->reqs[0]);
        MPI_Irecv(tmprecv, msglen, MPI_BYTE, recvfrom, 2703,
                  comm, &oob_req->reqs[1]);
    }
    probe = 0;
    do {
        MPI_Testall(2, oob_req->reqs, &completed, MPI_STATUS_IGNORE);
        probe++;
    } while (!completed && probe < probe_count);
    if (!completed) {
        return XCCL_INPROGRESS;
    }
    return XCCL_OK;
}

static xccl_status_t oob_allgather_free(void *req)
{
    free(req);
    return XCCL_OK;
}

static xccl_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   int my_rank, xccl_ep_range_t range,
                                   void *oob_coll_ctx, void **req)
{
    xccl_test_oob_allgather_req_t *oob_req = malloc(sizeof(*oob_req));
    oob_req->sbuf = sbuf;
    oob_req->rbuf = rbuf;
    oob_req->msglen = msglen;
    oob_req->range = range;
    oob_req->oob_coll_ctx = oob_coll_ctx;
    oob_req->my_rank = my_rank;
    oob_req->iter = 0;
    *req = oob_req;
    return oob_allgather_test(*req);
}

int xccl_mpi_create_comm_nb(MPI_Comm comm, xccl_team_h *team) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Create XCCL TEAM for comm world */
    xccl_team_config_t team_config = {
        .range     = {
            .type           = XCCL_EP_RANGE_STRIDED,
            .strided.start  = 0,
            .strided.stride = 1
        }
    };

    xccl_oob_collectives_t oob = {
        .allgather    = oob_allgather,
        .req_test     = oob_allgather_test,
        .req_free     = oob_allgather_free,
        .coll_context = (void*)comm,
        .rank = rank,
        .size = size
    };
    XCCL_CHECK(xccl_team_create_post(team_ctx, &team_config, oob, team));
    return 0;
}

int xccl_mpi_create_comm(MPI_Comm comm, xccl_team_h *team) {
    xccl_mpi_create_comm_nb(comm, team);
    while (XCCL_INPROGRESS == xccl_team_create_test(*team)) {;};
}

int xccl_mpi_test_init(int argc, char **argv,
                       xccl_collective_cap_t coll_types) {
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
        .coll_types = coll_types,
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
                .req_test     = oob_allgather_test,
                .req_free     = oob_allgather_free,
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
    xccl_mpi_create_comm(MPI_COMM_WORLD, &xccl_world_team);
    return 0;
}

int xccl_mpi_test_finalize(void) {
    XCCL_CHECK(xccl_team_destroy(xccl_world_team));
    XCCL_CHECK(xccl_cleanup(team_ctx));
    MPI_Finalize();
    return 0;
}
