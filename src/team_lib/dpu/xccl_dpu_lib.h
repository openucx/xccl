/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_DPU_H_
#define XCCL_TEAM_LIB_DPU_H_

#include "xccl_team_lib.h"
#include <ucp/api/ucp.h>

typedef struct xccl_team_lib_dpu_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_dpu_config_t;

typedef struct xccl_tl_dpu_context_config {
    xccl_tl_context_config_t super;
    char                     *server_hname;
    uint32_t                 server_port;
    uint32_t                 use_dpu;
    char                     *host_dpu_list;
} xccl_tl_dpu_context_config_t;

typedef struct xccl_team_lib_dpu {
    xccl_team_lib_t            super;
    xccl_team_lib_dpu_config_t config;
} xccl_team_lib_dpu_t;

extern xccl_team_lib_dpu_t xccl_team_lib_dpu;

#define xccl_team_dpu_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_dpu.config.super.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_dpu_error(_fmt, ...)        xccl_team_dpu_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_dpu_warn(_fmt, ...)         xccl_team_dpu_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_dpu_info(_fmt, ...)         xccl_team_dpu_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_dpu_debug(_fmt, ...)        xccl_team_dpu_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_dpu_trace(_fmt, ...)        xccl_team_dpu_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_dpu_trace_req(_fmt, ...)    xccl_team_dpu_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_dpu_trace_data(_fmt, ...)   xccl_team_dpu_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_dpu_trace_async(_fmt, ...)  xccl_team_dpu_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_dpu_trace_func(_fmt, ...)   xccl_team_dpu_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_dpu_trace_poll(_fmt, ...)   xccl_team_dpu_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)


typedef struct xccl_dpu_context {
    xccl_tl_context_t super;
    ucp_context_h     ucp_context;
    ucp_worker_h      ucp_worker;
    ucp_ep_h          ucp_ep;
} xccl_dpu_context_t;

typedef struct xccl_dpu_team {
    xccl_tl_team_t super;
    uint32_t       coll_id;
    uint32_t       ctrl_seg[1];
    ucp_mem_h      ctrl_seg_memh;
    uint64_t       rem_ctrl_seg;
    ucp_rkey_h     rem_ctrl_seg_key;
    uint64_t       rem_data_in;
    ucp_rkey_h     rem_data_in_key;
    uint64_t       rem_data_out;
    ucp_rkey_h     rem_data_out_key;
} xccl_dpu_team_t;

typedef struct dpu_sync_t {
    unsigned int itt;
    unsigned int dtype;
    unsigned int op;
    unsigned int len;
} dpu_sync_t;

typedef struct xccl_dpu_coll_req {
    xccl_tl_coll_req_t  super;
    xccl_tl_team_t      *team;
    xccl_coll_op_args_t args;
    dpu_sync_t          sync;
} xccl_dpu_coll_req_t;

#endif
