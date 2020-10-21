/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_UCX_TEAM_H_
#define XCCL_UCX_TEAM_H_
#include "xccl_ucx_lib.h"

typedef struct xccl_ucx_team_t {
    xccl_tl_team_t  super;
    uint16_t        ctx_id;
    uint16_t        seq_num;
    int             max_addrlen;
    ucp_ep_h        *ucp_eps;
    xccl_ep_range_t range;
    void            *nb_create_req;
} xccl_ucx_team_t;

xccl_status_t xccl_ucx_team_create_post(xccl_tl_context_t *context,
                                        xccl_team_params_t *params,
                                        xccl_team_t *base_team,
                                        xccl_tl_team_t **team);
xccl_status_t xccl_ucx_team_create_test(xccl_tl_team_t *team);
xccl_status_t xccl_ucx_team_destroy(xccl_tl_team_t *team);
#endif
