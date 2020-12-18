/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_MKEYS_H
#define XCCL_MHBA_MKEYS_H
#include "api/xccl.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

typedef struct xccl_mhba_context  xccl_mhba_context_t;
typedef struct xccl_mhba_node     xccl_mhba_node_t;
typedef struct xccl_mhba_coll_req xccl_mhba_coll_req_t;
typedef struct xccl_mhba_team     xccl_mhba_team_t;

#define UMR_CQ_SIZE 8 //todo check

xccl_status_t xccl_mhba_init_umr(xccl_mhba_context_t *ctx,
                                 xccl_mhba_node_t    *node);

xccl_status_t xccl_mhba_init_mkeys(xccl_mhba_team_t *team);

xccl_status_t xccl_mhba_populate_send_recv_mkeys(xccl_mhba_team_t     *team,
                                                 xccl_mhba_coll_req_t *req);

xccl_status_t xccl_mhba_update_mkeys_entries(xccl_mhba_node_t     *node,
                                             xccl_mhba_coll_req_t *req);

xccl_status_t xccl_mhba_destroy_umr(xccl_mhba_node_t *node);

xccl_status_t xccl_mhba_destroy_mkeys(xccl_mhba_node_t *node, int error_mode);
#endif //XCCL_MHBA_MKEYS_H
