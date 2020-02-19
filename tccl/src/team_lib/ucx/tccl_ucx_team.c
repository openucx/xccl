/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "tccl_ucx_context.h"
#include "tccl_ucx_team.h"
#include "tccl_ucx_ep.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

tccl_status_t tccl_ucx_team_create_post(tccl_context_t *context, tccl_team_config_t *config,
                                      tccl_oob_collectives_t oob, tccl_team_t **team)
{
    //TODO need to make this non blocking + team_ucx_wait
    tccl_status_t status      = TCCL_OK;
    tccl_team_lib_ucx_context_t *ctx =
        tccl_derived_of(context, tccl_team_lib_ucx_context_t);
    int max_cid = 0, max_addrlen = 0, size = config->team_size,
        rank = config->team_rank;
    tccl_ucx_team_t *ucx_team;
    int *tmp;
    int local_addrlen, i, sbuf[2];
    char* addr_array;

    ucx_team = (tccl_ucx_team_t*)malloc(sizeof(tccl_ucx_team_t));
    TCCL_TEAM_SUPER_INIT(ucx_team->super, context, config, oob);

    local_addrlen            = (int)ctx->ucp_addrlen;
    tmp                      = (int*)malloc(size*sizeof(int)*2);
    sbuf[0]                  = local_addrlen;
    sbuf[1]                  = ctx->next_cid;
    oob.allgather(sbuf, tmp, 2*sizeof(int), oob.coll_context);
    for (i=0; i<size; i++) {
        if (tmp[2*i] > max_addrlen) max_addrlen = tmp[2*i];
        if (tmp[2*i+1] > max_cid)   max_cid     = tmp[2*i+1];
    }
    free(tmp);

    ucx_team->ctx_id  = (uint16_t)max_cid; // TODO check overflow
    ucx_team->seq_num = 0;
    ctx->next_cid     = max_cid + 1; // this is only a tmp solution to max_cid
                                     // need another alg for cid allocatoin or
                                     // and interface to get from user
    addr_array        = (char*)malloc(size*max_addrlen);
    oob.allgather(ctx->worker_address, addr_array, max_addrlen, oob.coll_context);

    if (!ctx->ucp_eps) {
        ucx_team->ucp_eps = (ucp_ep_h*)calloc(size, sizeof(ucp_ep_h));
    } else {
        ucx_team->ucp_eps = NULL;
    }

    for (i=0; i<size; i++) {
        if (TCCL_OK != (status = connect_ep(ctx, ucx_team, config,
                                           addr_array, max_addrlen, i))) {
            goto cleanup;
        }
    }
    *team = &ucx_team->super;
cleanup:
    free(addr_array);
    return TCCL_OK;
}

tccl_status_t tccl_ucx_team_destroy(tccl_team_t *team)
{
    tccl_ucx_team_t             *ucx_team = tccl_derived_of(team, tccl_ucx_team_t);
    tccl_team_lib_ucx_context_t *ctx      = tccl_derived_of(team->ctx, tccl_team_lib_ucx_context_t);
    void *tmp;

    if (ucx_team->ucp_eps) {
        close_eps(ucx_team->ucp_eps, team->oob.size, ctx->ucp_worker);
        tmp = malloc(team->oob.size);
        team->oob.allgather(tmp, tmp, 1, team->oob.coll_context);
        free(tmp);
        free(ucx_team->ucp_eps);
    }
    free(ucx_team);
    return TCCL_OK;
}
