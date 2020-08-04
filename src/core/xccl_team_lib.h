/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_TEAM_LIB_H_
#define XCCL_TEAM_LIB_H_

#include "config.h"
#include "api/xccl.h"
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include <utils/xccl_log.h>
#include <assert.h>
#include <string.h>
#include "xccl_schedule.h"

typedef struct xccl_tl_context  xccl_tl_context_t;
typedef struct xccl_tl_team     xccl_tl_team_t;
typedef struct xccl_tl_coll_req xccl_tl_coll_req_t;
typedef struct xccl_team_lib* xccl_team_lib_h;

typedef struct xccl_team_lib_config {
    /* Log level above which log messages will be printed */
    ucs_log_component_config_t log_component;
    
    /* Team library priority */
    int                        priority;
} xccl_team_lib_config_t;

extern ucs_config_field_t xccl_team_lib_config_table[];

typedef struct xccl_tl_context_config {
    char  *env_prefix;
} xccl_tl_context_config_t;

extern ucs_config_field_t xccl_tl_context_config_table[];

typedef struct xccl_tl_mem_handle*   xccl_tl_mem_h;
typedef struct xccl_team_lib {
    char*                          name;
    xccl_tl_id_t                   id;
    int                            priority;
    xccl_lib_params_t              params;
    xccl_context_create_mode_t     ctx_create_mode;
    void*                          dl_handle;
    ucs_config_global_list_entry_t team_lib_config;
    ucs_config_global_list_entry_t tl_context_config;
    xccl_status_t              (*team_lib_open)(xccl_team_lib_h self,
                                                xccl_team_lib_config_t *config);
    void                       (*team_lib_close)(xccl_team_lib_h self);
    xccl_status_t              (*team_lib_query)(xccl_team_lib_h lib,
                                                 xccl_tl_attr_t *attr);
    xccl_status_t              (*team_context_create)(xccl_team_lib_h lib,
                                                      xccl_context_params_t *params,
                                                      xccl_tl_context_config_t *config,
                                                      xccl_tl_context_t **team_context);
    xccl_status_t              (*team_context_progress)(xccl_tl_context_t *team_context);
    xccl_status_t              (*team_context_destroy)(xccl_tl_context_t *team_context);
    xccl_status_t              (*team_create_post)(xccl_tl_context_t *team_ctx,
                                                   xccl_team_params_t *params,
                                                   xccl_tl_team_t **team);
    xccl_status_t              (*team_create_test)(xccl_tl_team_t *team_ctx);
    xccl_status_t              (*team_destroy)(xccl_tl_team_t *team);
    xccl_status_t              (*collective_init)(xccl_coll_op_args_t *coll_args,
                                                  xccl_tl_coll_req_t **request,
                                                  xccl_tl_team_t *team);
    xccl_status_t              (*collective_post)(xccl_tl_coll_req_t *request);
    xccl_status_t              (*collective_wait)(xccl_tl_coll_req_t *request);
    xccl_status_t              (*collective_test)(xccl_tl_coll_req_t *request);
    xccl_status_t              (*collective_finalize)(xccl_tl_coll_req_t *request);
    xccl_status_t              (*global_mem_map_start)(xccl_tl_team_t *team,
                                                       xccl_mem_map_params_t params,
                                                       xccl_tl_mem_h *memh_p);
    xccl_status_t              (*global_mem_map_test)(xccl_tl_mem_h memh_p);
    xccl_status_t              (*global_mem_unmap)(xccl_tl_mem_h memh_p);
} xccl_team_lib_t;

typedef struct xccl_progress_queue xccl_progress_queue_t;
typedef struct xccl_tl_context {
    xccl_team_lib_t       *lib;
    xccl_context_params_t params;
    xccl_progress_queue_t *pq;
} xccl_tl_context_t;

typedef struct xccl_tl_team {
    xccl_tl_context_t  *ctx;
    xccl_team_params_t params;
} xccl_tl_team_t;

typedef struct xccl_tl_coll_req {
    xccl_team_lib_t *lib;
} xccl_tl_coll_req_t;

static inline void
xccl_oob_allgather_nb(void *sbuf, void* rbuf, size_t len,
                      xccl_oob_collectives_t *oob, void **req)
{
    xccl_ep_range_t r = {
        .type = XCCL_EP_RANGE_UNDEFINED,
    };
    oob->allgather(sbuf, rbuf, len, 0, r, oob->coll_context, req);
}

static inline void
xccl_oob_allgather(void *sbuf, void* rbuf, size_t len, xccl_oob_collectives_t *oob)
{
    void *req;
    xccl_oob_allgather_nb(sbuf, rbuf, len, oob, &req);
    while (XCCL_INPROGRESS == oob->req_test(req)) {;}
    oob->req_free(req);
}

typedef struct xccl_local_proc_info {
    unsigned long node_hash;
    int           socketid; //if process is bound to a socket
    int           pid;
} xccl_local_proc_info_t;

xccl_local_proc_info_t* xccl_local_process_info();

#define XCCL_TEAM_SUPER_INIT(_team, _ctx, _params) do {                   \
        (_team).ctx = (_ctx);                                             \
        memcpy(&((_team).params), (_params), sizeof(xccl_team_params_t)); \
    }while(0)

#define XCCL_CONTEXT_SUPER_INIT(_ctx, _lib, _params) do {                 \
        (_ctx).lib = (_lib);                                              \
        memcpy(&((_ctx).params), (_params), sizeof(xccl_context_params_t));  \
    }while(0)

#endif
