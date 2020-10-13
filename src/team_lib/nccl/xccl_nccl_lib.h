/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_NCCL_H_
#define XCCL_TEAM_LIB_NCCL_H_

#include <nccl.h>
#include <cuda.h>
#include "xccl_team_lib.h"

typedef struct xccl_team_lib_nccl_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_nccl_config_t;

typedef struct xccl_tl_nccl_context_config {
    xccl_tl_context_config_t super;
    char                     *device;
} xccl_tl_nccl_context_config_t;

typedef struct xccl_team_lib_nccl {
    xccl_team_lib_t             super;
    xccl_team_lib_nccl_config_t config;
} xccl_team_lib_nccl_t;

extern xccl_team_lib_nccl_t xccl_team_lib_nccl;

#define xccl_team_nccl_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_nccl.config.super.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_nccl_error(_fmt, ...)       xccl_team_nccl_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_nccl_warn(_fmt, ...)        xccl_team_nccl_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_nccl_info(_fmt, ...)        xccl_team_nccl_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_nccl_debug(_fmt, ...)       xccl_team_nccl_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_nccl_trace(_fmt, ...)       xccl_team_nccl_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_nccl_trace_req(_fmt, ...)   xccl_team_nccl_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_nccl_trace_data(_fmt, ...)  xccl_team_nccl_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_nccl_trace_async(_fmt, ...) xccl_team_nccl_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_nccl_trace_func(_fmt, ...)  xccl_team_nccl_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_nccl_trace_poll(_fmt, ...)  xccl_team_nccl_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

typedef struct xccl_nccl_context {
    xccl_tl_context_t super;
} xccl_nccl_context_t;

typedef struct xccl_nccl_team {
    xccl_tl_team_t super;
    ncclComm_t     nccl_comm;
    cudaStream_t   stream;
} xccl_nccl_team_t;

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if(cudaSuccess != e) {                            \
    nccl_ucx_error("CUDA error %s:%d '%d' %s\n",    \
        __FILE__,__LINE__, e, cudaGetErrorName(e)); \
    return XCCL_ERR_NO_MESSAGE;                     \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                           \
  ncclResult_t e = cmd;                               \
  if(ncclSuccess != e) {                              \
    nccl_ucx_error("NCCL error %s:%d '%d'\n",         \
        __FILE__,__LINE__, e, ncclGetErrorString(e)); \
    return XCCL_ERR_NO_MESSAGE;                       \
  }                                                   \
} while(0)

#endif
