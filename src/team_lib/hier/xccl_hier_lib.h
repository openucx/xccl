/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_HIER_H_
#define XCCL_TEAM_LIB_HIER_H_

#include "xccl_team_lib.h"

typedef struct xccl_team_lib_hier_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_hier_config_t;

typedef struct xccl_tl_hier_context_config {
    xccl_tl_context_config_t super;
    ucs_config_names_array_t devices;
    int                      enable_sharp;
    int                      enable_shmseg;
    int                      enable_vmc;
    unsigned                 bcast_pipeline_depth;
    int                      bcast_sm_get;
    int                      node_leader_rank_id;
    size_t                   bcast_sm_get_thresh;
    size_t                   bcast_pipeline_thresh;
} xccl_tl_hier_context_config_t;

typedef struct xccl_team_lib_hier {
    xccl_team_lib_t             super;
    xccl_team_lib_hier_config_t config;
    xccl_lib_h                  tl_lib;
} xccl_team_lib_hier_t;
extern xccl_team_lib_hier_t xccl_team_lib_hier;

#define xccl_team_hier_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_hier.config.super.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_hier_error(_fmt, ...)        xccl_team_hier_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_hier_warn(_fmt, ...)         xccl_team_hier_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_hier_info(_fmt, ...)         xccl_team_hier_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_hier_debug(_fmt, ...)        xccl_team_hier_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_hier_trace(_fmt, ...)        xccl_team_hier_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_hier_trace_req(_fmt, ...)    xccl_team_hier_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_hier_trace_data(_fmt, ...)   xccl_team_hier_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_hier_trace_async(_fmt, ...)  xccl_team_hier_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_hier_trace_func(_fmt, ...)   xccl_team_hier_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_hier_trace_poll(_fmt, ...)   xccl_team_hier_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

#endif
