/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_MRAIL_H_
#define XCCL_TEAM_LIB_MRAIL_H_

#include <api/xccl.h>
#include <xccl_team_lib.h>

#define MAX_TLS_NUMBER 16

typedef struct xccl_team_lib_mrail_config {
    xccl_team_lib_config_t super;
    xccl_tl_id_t           replicated_tl_id;
    unsigned               replicas_num;
    unsigned               threads_num;
    unsigned               thread_poll_cnt;
} xccl_team_lib_mrail_config_t;

typedef struct xccl_tl_mrail_context_config {
    xccl_tl_context_config_t super;
    ucs_config_names_array_t devices;
} xccl_tl_mrail_context_config_t;

typedef struct xccl_mrail_progress_thread {
    ucs_list_link_t list;
    pthread_t       tid;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    unsigned        poll_cnt;
    int             close;
} xccl_mrail_progress_thread_t;

typedef struct xccl_mrail_progress_request {
    ucs_list_link_t list;
    xccl_context_h  ctx;
    xccl_coll_req_h req;
    xccl_status_t   completed;
} xccl_mrail_progress_request_t;

typedef struct xccl_team_lib_mrail {
    xccl_team_lib_t              super;
    xccl_team_lib_mrail_config_t config;
    xccl_mrail_progress_thread_t threads[MAX_TLS_NUMBER];
} xccl_team_lib_mrail_t;
extern xccl_team_lib_mrail_t xccl_team_lib_mrail;

typedef struct xccl_mrail_context {
    xccl_tl_context_t super;
/* tls array holds n_tls pointers to some other team library */
    xccl_context_h    tls[MAX_TLS_NUMBER];
    xccl_lib_h        tl;
/* number of times team library is replicated, 
 * typically equal to number of HW resources available
 */
    size_t            n_tls;
} xccl_mrail_context_t;

//TODO: how many teams allowed per context? Do we need more than 1 team?
typedef struct xccl_mrail_team {
    xccl_tl_team_t super;
/* teams array holds n_teams pointers to some other team library teams */
    xccl_team_h    teams[MAX_TLS_NUMBER];
    size_t         n_teams;
} xccl_mrail_team_t;

typedef struct xccl_mrail_coll_req {
    xccl_tl_coll_req_t super;
    xccl_mrail_team_t  team;
    xccl_mrail_progress_request_t reqs[MAX_TLS_NUMBER];
    size_t             n_reqs;
} xccl_mrail_coll_req_t;

#define xccl_team_mrail_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_mrail.config.super.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_mrail_error(_fmt, ...)        xccl_team_mrail_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_mrail_warn(_fmt, ...)         xccl_team_mrail_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_mrail_info(_fmt, ...)         xccl_team_mrail_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_mrail_debug(_fmt, ...)        xccl_team_mrail_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_mrail_trace(_fmt, ...)        xccl_team_mrail_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_mrail_trace_req(_fmt, ...)    xccl_team_mrail_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_mrail_trace_data(_fmt, ...)   xccl_team_mrail_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_mrail_trace_async(_fmt, ...)  xccl_team_mrail_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_mrail_trace_func(_fmt, ...)   xccl_team_mrail_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_mrail_trace_poll(_fmt, ...)   xccl_team_mrail_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

#endif
