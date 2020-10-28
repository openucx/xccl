/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_MHBA_MKEYS_H
#define XCCL_MHBA_MKEYS_H

typedef struct xccl_mhba_context xccl_mhba_context_t;
typedef struct xccl_mhba_node xccl_mhba_node_t;

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include "api/xccl.h"

#define UMR_CQ_SIZE 2 //todo check

xccl_status_t init_umr(xccl_mhba_context_t *ctx);

xccl_status_t init_mkeys(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node);

xccl_status_t update_mkeys(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node);

xccl_status_t destroy_umr(xccl_mhba_context_t *ctx);

xccl_status_t destroy_mkeys(xccl_mhba_node_t *node);


#endif //XCCL_MHBA_MKEYS_H
