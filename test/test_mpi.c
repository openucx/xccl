#include "test_mpi.h"
#include <assert.h>

xccl_team_h xccl_world_team;
static xccl_lib_h     lib;
xccl_context_h team_ctx;

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
    xccl_team_params_t team_params = {
        .field_mask         = XCCL_TEAM_PARAM_FIELD_EP_RANGE |
                              XCCL_TEAM_PARAM_FIELD_OOB,
        .range = {
            .type           = XCCL_EP_RANGE_STRIDED,
            .strided.start  = 0,
            .strided.stride = 1
        },

        .oob   = {
            .allgather      = oob_allgather,
            .req_test       = oob_allgather_test,
            .req_free       = oob_allgather_free,
            .coll_context   = (void*)comm,
            .rank           = rank,
            .size           = size
        }
    };

    XCCL_CHECK(xccl_team_create_post(team_ctx, &team_params, team));
    return 0;
}

int xccl_mpi_create_comm(MPI_Comm comm, xccl_team_h *team) {
    xccl_mpi_create_comm_nb(comm, team);
    while (XCCL_INPROGRESS == xccl_team_create_test(*team)) {;};
}

int xccl_mpi_test_init(int argc, char **argv,
                       xccl_collective_cap_t coll_types) {
    char     *var;
    int      rank, size;
    char     *tl, *saveptr;
    uint64_t tls = 0;
    uint64_t i;
    int      j;
    xccl_tl_id_t *tl_ids;
    unsigned     tl_count, provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Init xccl library */
    var = getenv("XCCL_TEST_TLS");
    if (var) {
        tls = xccl_tls_str_to_bitmap(var);
    }
    else {
        tls = XCCL_TL_ALL;
    }

    xccl_lib_params_t lib_params = {
        .field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE |
                      XCCL_LIB_PARAM_FIELD_COLL_TYPES,
        .team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                      XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
        .coll_types = coll_types,
    };

    xccl_lib_config_t *cfg;


    /* Init xccl context for a specified XCCL_TEST_TLS */
    xccl_context_params_t ctx_params = {
        .field_mask      = XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE |
                           XCCL_CONTEXT_PARAM_FIELD_OOB |
                           XCCL_CONTEXT_PARAM_FIELD_TEAM_COMPLETION_TYPE |
                           XCCL_CONTEXT_PARAM_FIELD_TLS,
        .thread_mode     = XCCL_THREAD_MODE_SINGLE,
        .completion_type = XCCL_TEAM_COMPLETION_TYPE_BLOCKING,
        .oob = {
            .allgather    = oob_allgather,
            .req_test     = oob_allgather_test,
            .req_free     = oob_allgather_free,
            .coll_context = (void*)MPI_COMM_WORLD,
            .rank         = rank,
            .size         = size
        },
        .tls              = tls,
    };
    XCCL_CHECK(xccl_lib_init(&lib_params, cfg, &lib));

    XCCL_CHECK(xccl_get_tl_list(lib, &tl_ids, &tl_count));
    xccl_free_tl_list(tl_ids);

    xccl_context_config_t *ctx_config;
    XCCL_CHECK(xccl_context_config_read(lib, NULL, NULL, &ctx_config));
    XCCL_CHECK(xccl_context_create(lib, &ctx_params, ctx_config, &team_ctx));
    xccl_context_config_release(ctx_config);
    xccl_mpi_create_comm(MPI_COMM_WORLD, &xccl_world_team);
    return 0;
}

void xccl_mpi_test_finalize(void) {
    xccl_team_destroy(xccl_world_team);
    xccl_context_destroy(team_ctx);
    xccl_lib_cleanup(lib);
    MPI_Finalize();
}

void xccl_mpi_test_progress(void)
{
    xccl_context_progress(team_ctx);
}
