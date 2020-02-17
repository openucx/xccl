/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_UCX_TEAM_H_
#define TCCL_UCX_TEAM_H_
#include "tccl_ucx_lib.h"

typedef struct tccl_ucx_team_t {
    tccl_team_t   super;
    uint16_t     ctx_id;
    uint16_t     seq_num;
    ucp_ep_h*    ucp_eps;
} tccl_ucx_team_t;

tccl_status_t tccl_ucx_team_create_post(tccl_team_context_t *context, tccl_team_config_t *config,
                                      tccl_oob_collectives_t oob, tccl_team_t **team);
tccl_status_t tccl_ucx_team_destroy(tccl_team_t *team);

#endif
