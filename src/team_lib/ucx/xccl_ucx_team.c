/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "xccl_ucx_context.h"
#include "xccl_ucx_team.h"
#include "xccl_ucx_ep.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

struct xccl_ucx_nb_create_req {
    int phase;
    void *scratch;
    void *allgather_req;
};

xccl_status_t xccl_ucx_team_create_post(xccl_tl_context_t *context, xccl_team_config_t *config,
                                        xccl_oob_collectives_t oob, xccl_tl_team_t **team)
{
    xccl_status_t status      = XCCL_OK;
    xccl_team_lib_ucx_context_t *ctx =
        xccl_derived_of(context, xccl_team_lib_ucx_context_t);
    int max_cid = 0, max_addrlen = 0, size = oob.size,
        rank = oob.rank;
    xccl_ucx_team_t *ucx_team;
    int *tmp;
    int local_addrlen, i, sbuf[2];
    char* addr_array;
    struct xccl_ucx_nb_create_req *nb_req = malloc(sizeof(*nb_req));
    ucx_team = (xccl_ucx_team_t*)malloc(sizeof(xccl_ucx_team_t));
    XCCL_TEAM_SUPER_INIT(ucx_team->super, context, config, oob);
    nb_req->phase = 0;
    ucx_team->nb_create_req  = nb_req;
    ucx_team->range          = config->range;
    local_addrlen            = (int)ctx->ucp_addrlen;
    tmp                      = (int*)malloc(size*sizeof(int)*2);
    sbuf[0]                  = local_addrlen;
    sbuf[1]                  = ctx->next_cid;
    xccl_oob_allgather_nb(sbuf, tmp, 2*sizeof(int), &oob, &nb_req->allgather_req);
    nb_req->scratch = tmp;
    *team = &ucx_team->super;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_team_create_test(xccl_tl_team_t *team)
{
    xccl_status_t status      = XCCL_OK;
    xccl_team_lib_ucx_context_t *ctx =
        xccl_derived_of(team->ctx, xccl_team_lib_ucx_context_t);
    xccl_oob_collectives_t oob = team->oob;
    int max_cid = 0, size = oob.size,
        rank = oob.rank;
    xccl_ucx_team_t *ucx_team = xccl_derived_of(team, xccl_ucx_team_t);
    int *tmp;
    int local_addrlen, i, sbuf[2];
    char* addr_array;
    struct xccl_ucx_nb_create_req *nb_req =
        (struct xccl_ucx_nb_create_req *)ucx_team->nb_create_req;
    if (NULL == nb_req) {
        return XCCL_OK;
    } else if (XCCL_INPROGRESS == oob.req_test(nb_req->allgather_req)) {
        return XCCL_INPROGRESS;
    }
    oob.req_free(nb_req->allgather_req);

    switch (nb_req->phase) {
    case 0:
        tmp = (int*)nb_req->scratch;
        ucx_team->max_addrlen = 0;
        for (i=0; i<size; i++) {
            if (tmp[2*i] > ucx_team->max_addrlen) ucx_team->max_addrlen = tmp[2*i];
            if (tmp[2*i+1] > max_cid)   max_cid     = tmp[2*i+1];
        }
        free(tmp);

        ucx_team->ctx_id  = (uint16_t)max_cid; // TODO check overflow
        ucx_team->seq_num = 0;
        ctx->next_cid     = max_cid + 1; // this is only a tmp solution to max_cid
                                         // need another alg for cid allocatoin or
                                         // and interface to get from user
        addr_array        = (char*)malloc(size*ucx_team->max_addrlen);
        xccl_oob_allgather_nb(ctx->worker_address, addr_array,
                              ucx_team->max_addrlen, &oob, &nb_req->allgather_req);
        nb_req->phase = 1;
        nb_req->scratch = addr_array;
        return XCCL_INPROGRESS;
    case 1:
        addr_array = (char*)nb_req->scratch;
        if (!ctx->ucp_eps) {
            ucx_team->ucp_eps = (ucp_ep_h*)calloc(size, sizeof(ucp_ep_h));
        } else {
            ucx_team->ucp_eps = NULL;
        }

        for (i=0; i<size; i++) {
            if (XCCL_OK != (status = connect_ep(ctx, ucx_team,
                                                addr_array, ucx_team->max_addrlen, i))) {
                status = XCCL_ERR_NO_MESSAGE;
                goto cleanup;
            }
        }
        break;
    }

cleanup:
    free(addr_array);
    free(nb_req);
    ucx_team->nb_create_req = NULL;
    return status;
}

xccl_status_t xccl_ucx_team_destroy(xccl_tl_team_t *team)
{
    xccl_ucx_team_t             *ucx_team = xccl_derived_of(team, xccl_ucx_team_t);
    xccl_team_lib_ucx_context_t *ctx      = xccl_derived_of(team->ctx, xccl_team_lib_ucx_context_t);
    void *tmp;

    if (ucx_team->ucp_eps) {
        close_eps(ucx_team->ucp_eps, team->oob.size, ctx->ucp_worker);
        tmp = malloc(team->oob.size);
        xccl_oob_allgather(tmp, tmp, 1, &team->oob);
        free(tmp);
        free(ucx_team->ucp_eps);
    }
    free(ucx_team);
    return XCCL_OK;
}
