/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_MKEYS_H
#define XCCL_MHBA_MKEYS_H

typedef struct xccl_mhba_context xccl_mhba_context_t;
typedef struct xccl_mhba_node xccl_mhba_node_t;
typedef struct xccl_mhba_coll_req xccl_mhba_coll_req_t;

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include "api/xccl.h"


#define UMR_CQ_SIZE 2 //todo check

xccl_status_t xccl_mhba_init_umr(xccl_mhba_context_t *ctx);

xccl_status_t xccl_mhba_init_mkeys(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node, int team_size);

xccl_status_t xccl_mhba_populate_send_recv_mkeys(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node,int seq_num,int block_size,int team_size);

xccl_status_t
xccl_mhba_update_mkeys_entries(xccl_mhba_node_t *node, xccl_mhba_coll_req_t *req);

xccl_status_t xccl_mhba_destroy_umr(xccl_mhba_context_t *ctx);

xccl_status_t xccl_mhba_destroy_mkeys(xccl_mhba_node_t *node);


#endif //XCCL_MHBA_MKEYS_H
