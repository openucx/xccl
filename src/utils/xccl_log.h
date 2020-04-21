/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_UTILS_LOG_H_
#define XCCL_UTILS_LOG_H_

#include "config.h"
#include "xccl_global_opts.h"

extern xccl_config_t xccl_lib_global_config;
#define xccl_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_lib_global_config.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_error(_fmt, ...)        xccl_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_warn(_fmt, ...)         xccl_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_info(_fmt, ...)         xccl_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_debug(_fmt, ...)        xccl_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_trace(_fmt, ...)        xccl_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_trace_req(_fmt, ...)    xccl_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_trace_data(_fmt, ...)   xccl_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_trace_async(_fmt, ...)  xccl_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_trace_func(_fmt, ...)   xccl_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_trace_poll(_fmt, ...)   xccl_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

#endif
