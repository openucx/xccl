/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_MHBA_H_
#define XCCL_TEAM_LIB_MHBA_H_
#include "xccl_team_lib.h"
#include "xccl_mhba_mkeys.h"
#include "xccl_mhba_socket_comm.h"
#include "topo/xccl_topo.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include <ucs/memory/rcache.h>

#define MAX_OUTSTANDING_OPS 1 //todo change - according to limitations (52 top)
#define SEQ_INDEX(_seq_num) ((_seq_num) % MAX_OUTSTANDING_OPS)
#define SQUARED(_num) ((_num) * (_num))

typedef struct xccl_team_lib_mhba_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_mhba_config_t;

typedef struct xccl_tl_mhba_context_config {
    xccl_tl_context_config_t super;
    ucs_config_names_array_t devices;
    int                      ib_global;
    int                      transpose;
    int                      transpose_hw_limitations;
    size_t                   transpose_buf_size;
    int                      block_size;
} xccl_tl_mhba_context_config_t;

//todo add block_size config

typedef struct xccl_team_lib_mhba {
    xccl_team_lib_t             super;
    xccl_team_lib_mhba_config_t config;
} xccl_team_lib_mhba_t;

extern xccl_team_lib_mhba_t xccl_team_lib_mhba;

#define xccl_team_mhba_log_component(_level, _fmt, ...)                        \
    do {                                                                       \
        ucs_log_component(_level,                                              \
                          &xccl_team_lib_mhba.config.super.log_component,      \
                          _fmt, ##__VA_ARGS__);                                \
    } while (0)

#define xccl_mhba_error(_fmt, ...)                                             \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define xccl_mhba_warn(_fmt, ...)                                              \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define xccl_mhba_info(_fmt, ...)                                              \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define xccl_mhba_debug(_fmt, ...)                                             \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define xccl_mhba_trace(_fmt, ...)                                             \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)
#define xccl_mhba_trace_req(_fmt, ...)                                         \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ##__VA_ARGS__)
#define xccl_mhba_trace_data(_fmt, ...)                                        \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ##__VA_ARGS__)
#define xccl_mhba_trace_async(_fmt, ...)                                       \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ##__VA_ARGS__)
#define xccl_mhba_trace_func(_fmt, ...)                                        \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")",     \
                                 __FUNCTION__, ##__VA_ARGS__)
#define xccl_mhba_trace_poll(_fmt, ...)                                        \
    xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ##__VA_ARGS__)

#define MHBA_CTRL_SIZE 128 //todo change according to arch
#define MHBA_DATA_SIZE sizeof(struct mlx5dv_mr_interleaved)
#define MHBA_NUM_OF_BLOCKS_SIZE_BINS 8
#define MAX_TRANSPOSE_SIZE 8000 // HW transpose unit is limited to matrix size
#define MAX_MSG_SIZE 128 // HW transpose unit is limited to element size
#define MAX_STRIDED_ENTRIES 55 // from limit of NIC memory - Sergey Gorenko's email

typedef struct xccl_mhba_context {
    xccl_tl_context_t                  super;
    struct xccl_tl_mhba_context_config cfg;
    struct ibv_context                *ib_ctx;
    struct ibv_pd                     *ib_pd;
    int                                ib_port;
} xccl_mhba_context_t;

enum {
    XCCL_MHBA_NEED_SEND_MKEY_UPDATE = UCS_BIT(1),
    XCCL_MHBA_NEED_RECV_MKEY_UPDATE = UCS_BIT(2),
};

typedef struct xccl_mhba_ctrl {
    int      seq_num;
    uint8_t  mkey_cache_flag;
} xccl_mhba_ctrl_t;

typedef struct xccl_mhba_op {
    void               *ctrl;
    xccl_mhba_ctrl_t   *my_ctrl;
    void               *send_umr_data;
    void               *my_send_umr_data;
    void               *recv_umr_data;
    void               *my_recv_umr_data;
    struct mlx5dv_mkey *send_mkey;
    struct mlx5dv_mkey *recv_mkey;
} xccl_mhba_op_t;

/* This structure holds resources and data related to the "in-node"
   part of the algorithm. */
typedef struct xccl_mhba_node {
    int                  asr_rank;
    xccl_sbgp_t         *sbgp;
    void                *storage;
    xccl_mhba_op_t       ops[MAX_OUTSTANDING_OPS];
    struct mlx5dv_mkey  *team_send_mkey;
    struct mlx5dv_mkey  *team_recv_mkey;
    struct ibv_context  *shared_ctx;
    struct ibv_pd       *shared_pd;
    struct ibv_cq       *umr_cq;
    struct ibv_qp       *umr_qp;
    struct ibv_qp_ex    *umr_qpx;
    struct mlx5dv_qp_ex *umr_mlx5dv_qp_ex;
} xccl_mhba_node_t;

typedef struct xccl_mhba_reg {
    struct ibv_mr       *mr;
    ucs_rcache_region_t *region;
} xccl_mhba_reg_t;

static inline xccl_mhba_reg_t* xccl_rcache_ucs_get_reg_data(ucs_rcache_region_t *region) {
    return (xccl_mhba_reg_t *)((ptrdiff_t)region + sizeof(ucs_rcache_region_t));
}

typedef struct xccl_mhba_net {
    xccl_sbgp_t    *sbgp;
    int             net_size;
    int            *rank_map;
    struct ibv_qp **qps;
    struct ibv_cq  *cq;
    struct ibv_mr  *ctrl_mr;
    struct {
        void    *addr;
        uint32_t rkey;
    } * remote_ctrl;
    uint32_t       *rkeys;
    xccl_tl_team_t *ucx_team;
} xccl_mhba_net_t;

typedef struct xccl_mhba_team {
    xccl_tl_team_t       super;
    int                  transpose;
    int                  transpose_hw_limitations;
    uint64_t             max_msg_size;
    xccl_mhba_node_t     node;
    xccl_mhba_net_t      net;
    int                  sequence_number;
    int                  op_busy[MAX_OUTSTANDING_OPS];
    int                  cq_completions[MAX_OUTSTANDING_OPS];
    xccl_mhba_context_t *context;
    int                  blocks_sizes[MHBA_NUM_OF_BLOCKS_SIZE_BINS];
    int                  size;
    int                  previous_msg_size[MAX_OUTSTANDING_OPS];
    void*                previous_send_address[MAX_OUTSTANDING_OPS];
    void*                previous_recv_address[MAX_OUTSTANDING_OPS];
    uint64_t             dummy_atomic_buff;
    ucs_rcache_t        *rcache;
    int                  requested_block_size;
    struct ibv_mr       *dummy_bf_mr;
    struct ibv_wc       *work_completion;
    void                *transpose_buf;
    struct ibv_mr       *transpose_buf_mr;
} xccl_mhba_team_t;

xccl_status_t xccl_mhba_team_create_post(xccl_tl_context_t  *context,
                                         xccl_team_params_t *params,
                                         xccl_team_t        *base_team,
                                         xccl_tl_team_t    **team);
xccl_status_t xccl_mhba_team_create_test(xccl_tl_team_t *team);
xccl_status_t xccl_mhba_team_destroy(xccl_tl_team_t *team);

static inline xccl_mhba_ctrl_t*
xccl_mhba_get_ctrl(xccl_mhba_team_t *team, int op_index, int rank)
{
    return (xccl_mhba_ctrl_t*)((ptrdiff_t)team->node.ops[op_index].ctrl +
                               MHBA_CTRL_SIZE * rank);
}

static inline xccl_mhba_ctrl_t*
xccl_mhba_get_my_ctrl(xccl_mhba_team_t *team, int op_index)
{
    int my_rank = team->node.sbgp->group_rank;
    return xccl_mhba_get_ctrl(team, op_index, my_rank);
}
#endif
