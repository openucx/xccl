/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_XCCL_MHBA_SOCKET_COMM_H
#define XCCL_XCCL_MHBA_SOCKET_COMM_H

typedef struct xccl_mhba_node xccl_mhba_node_t;

#include "api/xccl.h"

xccl_status_t xccl_mhba_share_ctx_pd(xccl_mhba_team_t *team,
                                     const char       *sock_path);
xccl_status_t xccl_mhba_remove_shared_ctx_pd(xccl_mhba_team_t *team);

#endif //XCCL_XCCL_MHBA_SOCKET_COMM_H
