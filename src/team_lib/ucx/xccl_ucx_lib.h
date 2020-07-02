/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TEAM_LIB_UCX_H_
#define XCCL_TEAM_LIB_UCX_H_
#include "xccl_team_lib.h"
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>

typedef struct xccl_team_lib_ucx {
    xccl_team_lib_t super;
    ucs_log_component_config_t log_component;
} xccl_team_lib_ucx_t;

typedef struct xccl_team_lib_ucx_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_ucx_config_t;

typedef struct xccl_tl_ucx_context_config {
    xccl_tl_context_config_t super;
    ucs_config_names_array_t devices;
    unsigned                 barrier_kn_radix;
    unsigned                 bcast_kn_radix;
    unsigned                 reduce_kn_radix;
    unsigned                 allreduce_kn_radix;
    unsigned                 num_to_probe;
    unsigned                 alltoall_pairwise_chunk;
    int                      alltoall_pairwise_reverse;
} xccl_tl_ucx_context_config_t;

extern xccl_team_lib_ucx_t xccl_team_lib_ucx;

#define xccl_team_ucx_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_ucx.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_ucx_error(_fmt, ...)        xccl_team_ucx_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_ucx_warn(_fmt, ...)         xccl_team_ucx_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_ucx_info(_fmt, ...)         xccl_team_ucx_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_ucx_debug(_fmt, ...)        xccl_team_ucx_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_ucx_trace(_fmt, ...)        xccl_team_ucx_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_ucx_trace_req(_fmt, ...)    xccl_team_ucx_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_ucx_trace_data(_fmt, ...)   xccl_team_ucx_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_ucx_trace_async(_fmt, ...)  xccl_team_ucx_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_ucx_trace_func(_fmt, ...)   xccl_team_ucx_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_ucx_trace_poll(_fmt, ...)   xccl_team_ucx_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

typedef enum {
    XCCL_UCX_REQUEST_ACTIVE,
    XCCL_UCX_REQUEST_DONE,
} xccl_ucx_request_status_t;

typedef struct xccl_ucx_request_t {
    xccl_ucx_request_status_t status;
} xccl_ucx_request_t;

#define MAX_REQS 32

typedef struct xccl_ucx_collreq {
    xccl_tl_coll_req_t  super;
    xccl_coll_op_args_t args;
    ucs_memory_type_t   mem_type;
    xccl_tl_team_t      *team;
    xccl_status_t       complete;
    uint16_t            tag;
    xccl_status_t       (*start)(struct xccl_ucx_collreq* req);
    xccl_status_t       (*progress)(struct xccl_ucx_collreq* req);
    union {
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            int                phase;
            int                iteration;
            int                radix_mask_pow;
            int                active_reqs;
            int                radix;
            void               *scratch;
        } allreduce;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                step;
            void               *scratch;
        } reduce_linear;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                step;
        } fanin_linear;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                step;
        } fanout_linear;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                step;
        } bcast_linear;
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            int                active_reqs;
            int                radix;
            int                dist;
        } bcast_kn;
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            uint8_t            active_reqs;
            uint8_t            radix;
            uint8_t            phase;
            int                dist;
            int                max_dist;
            void               *scratch;
            void               *data_buf;
        } reduce_kn;
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            int                phase;
            int                iteration;
            int                radix_mask_pow;
            int                active_reqs;
            int                radix;
        } barrier;
        struct {
            xccl_ucx_request_t **reqs;
            int                n_sreqs;
            int                n_rreqs;
        } alltoall_pairwise;
        struct {
            xccl_ucx_request_t *reqs[2];
            void               *scratch;
            int                step;
        } alltoall_linear_shift;
        struct {
            xccl_ucx_request_t **reqs;
            int                n_sreqs;
            int                n_rreqs;
        } alltoallv_pairwise;

    };
} xccl_ucx_collreq_t;


#endif
