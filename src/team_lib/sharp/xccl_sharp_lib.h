/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TEAM_LIB_SHARP_H_
#define XCCL_TEAM_LIB_SHARP_H_
#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>
#include "xccl_team_lib.h"

#define XCCL_SHARP_REG_BUF_SIZE 1024
#define XCCL_SHARP_REG_BUF_NUM  10

typedef struct xccl_team_lib_sharp {
    xccl_team_lib_t            super;
    ucs_log_component_config_t log_component;
} xccl_team_lib_sharp_t;

typedef struct xccl_team_lib_sharp_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_sharp_config_t;

extern xccl_team_lib_sharp_t xccl_team_lib_sharp;

#define xccl_team_sharp_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_sharp.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_sharp_error(_fmt, ...)        xccl_team_sharp_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_sharp_warn(_fmt, ...)         xccl_team_sharp_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_sharp_info(_fmt, ...)         xccl_team_sharp_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_sharp_debug(_fmt, ...)        xccl_team_sharp_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_sharp_trace(_fmt, ...)        xccl_team_sharp_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_sharp_trace_req(_fmt, ...)    xccl_team_sharp_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_sharp_trace_data(_fmt, ...)   xccl_team_sharp_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_sharp_trace_async(_fmt, ...)  xccl_team_sharp_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_sharp_trace_func(_fmt, ...)   xccl_team_sharp_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_sharp_trace_poll(_fmt, ...)   xccl_team_sharp_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

typedef struct xccl_sharp_context {
    xccl_tl_context_t             super;
    struct sharp_coll_context *sharp_context;
} xccl_sharp_context_t;

typedef struct xccl_sharp_buf {
    void *buf;
    void *mr;
    void *orig_src_buf;
    void *orig_dst_buf;
    int   used;
} xccl_sharp_buf_t;

typedef struct xccl_sharp_team {
    xccl_tl_team_t          super;
    struct sharp_coll_comm *sharp_comm;
    xccl_sharp_buf_t        bufs[XCCL_SHARP_REG_BUF_NUM];
} xccl_sharp_team_t;
#endif
