/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "server_xccl.h"
#include <assert.h>

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

int xccl_mpi_create_team_nb(dpu_xccl_comm_t *comm) {
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
            .coll_context   = (void*)MPI_COMM_WORLD,
            .rank           = comm->g->rank,
            .size           = comm->g->size
        }
    };

    XCCL_CHECK(xccl_team_create_post(comm->ctx, &team_params, &comm->team));
    return 0;
}

int xccl_mpi_create_team(dpu_xccl_comm_t *comm) {
    xccl_mpi_create_team_nb(comm);
    while (XCCL_INPROGRESS == xccl_team_create_test(comm->team)) {;};
}

int dpu_xccl_init(int argc, char **argv, dpu_xccl_global_t *g)
{
    char *var;
    xccl_tl_id_t *tl_ids;
    unsigned tl_count;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g->size);

    /* Init xccl library */
    var = getenv(DPU_XCCL_TLS);
    if (var) {
        g->tls = xccl_tls_str_to_bitmap(var);
    }
    else {
        g->tls = XCCL_TL_ALL;
    }

    xccl_lib_params_t lib_params = {
        .field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE |
                      XCCL_LIB_PARAM_FIELD_COLL_TYPES,
        .team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                      XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
        /* TODO: support more collectives */
        .coll_types = XCCL_COLL_CAP_ALLREDUCE,
    };

    XCCL_CHECK(xccl_lib_init(&lib_params, NULL, &g->lib));
    XCCL_CHECK(xccl_get_tl_list(g->lib, &tl_ids, &tl_count));
    xccl_free_tl_list(tl_ids);
    return XCCL_OK;
}

int dpu_xccl_alloc_team(dpu_xccl_global_t *g, dpu_xccl_comm_t *comm)
{
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
            .rank         = g->rank,
            .size         = g->size
        },
        .tls              = g->tls,
    };
    xccl_context_config_t *ctx_config;
    XCCL_CHECK(xccl_context_config_read(g->lib, NULL, NULL, &ctx_config));
    XCCL_CHECK(xccl_context_create(g->lib, &ctx_params, ctx_config, &comm->ctx));
    xccl_context_config_release(ctx_config);
    comm->g = g;
    xccl_mpi_create_team(comm);
    return XCCL_OK;
}

int dpu_xccl_free_team(dpu_xccl_global_t *g, dpu_xccl_comm_t *team)
{
    xccl_team_destroy(team->team);
    xccl_context_destroy(team->ctx);
}

void dpu_xccl_finalize(dpu_xccl_global_t *g) {
    xccl_lib_cleanup(g->lib);
    MPI_Finalize();
}

void dpu_xccl_progress(dpu_xccl_comm_t *comm)
{
    xccl_context_progress(comm->ctx);
}
