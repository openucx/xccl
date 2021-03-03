#ifndef XCCL_TEAM_LIB_MPOD_H_
#define XCCL_TEAM_LIB_MPOD_H_

#include "xccl_lib.h"
#include "xccl_team_lib.h"
#include "api/xccl_tls.h"
#include <xccl_ucs.h>
#include "uthash.h"
#include "mem_component.h"

#define xccl_team_mpod_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_mpod.config.super.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_mpod_err_pop(err, fn_fail) do { if (err != XCCL_OK) goto fn_fail; } while (0)
#define xccl_mpod_error(_fmt, ...)       xccl_team_mpod_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_mpod_warn(_fmt, ...)        xccl_team_mpod_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_mpod_info(_fmt, ...)        xccl_team_mpod_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_mpod_debug(_fmt, ...)       xccl_team_mpod_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_mpod_trace(_fmt, ...)       xccl_team_mpod_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_mpod_trace_req(_fmt, ...)   xccl_team_mpod_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_mpod_trace_data(_fmt, ...)  xccl_team_mpod_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_mpod_trace_async(_fmt, ...) xccl_team_mpod_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_mpod_trace_func(_fmt, ...)  xccl_team_mpod_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_mpod_trace_poll(_fmt, ...)  xccl_team_mpod_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

typedef struct xccl_team_lib_mpod_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_mpod_config_t;

typedef struct xccl_tl_mpod_context_config {
    xccl_tl_context_config_t super;
} xccl_tl_mpod_context_config_t;

typedef struct xccl_team_lib_mpod {
    xccl_team_lib_t super;
    xccl_team_lib_mpod_config_t config;
} xccl_team_lib_mpod_t;

typedef struct xccl_mpod_context {
    xccl_tl_context_t super;

    struct {
        xccl_team_lib_t *nccl;
        xccl_team_lib_t *ucx;
    } lib;

    struct {
        xccl_tl_context_t *nccl;
        xccl_tl_context_t *ucx_slice;
        xccl_tl_context_t *ucx_flat;
    } context;
} xccl_mpod_context_t;

typedef struct xccl_mpod_team {
    xccl_tl_team_t super;

    int pod_size;
    int num_pods;
    int pod_id;
    int slice_id;

    xccl_mpod_context_t *context;
    xccl_team_params_t user_params;
    xccl_oob_collectives_t user_oob_coll;

    struct {
        xccl_tl_team_t *nccl;
        xccl_tl_team_t *ucx_slice;
        xccl_tl_team_t *ucx_flat;
    } team;

    enum {
        UCX_TEAM_STATE__SLICE_TEAM_INITIATED,
        UCX_TEAM_STATE__FLAT_TEAM_INITIATED,
        UCX_TEAM_STATE__FLAT_TEAM_CREATED,
    } ucx_team_state;
} xccl_mpod_team_t;

#define XCCL_MPOD_MAX_NCCL_PHASES  (2)

struct xccl_mpod_coll_req;
typedef struct {
    xccl_tl_coll_req_t *r;
    struct xccl_mpod_coll_req *mpod_req;
    UT_hash_handle hh;
} xccl_mpod_nccl_req_s;

typedef struct {
    void *buf;
    UT_hash_handle hh;
} xccl_mpod_buf_s;

typedef struct {
    int phase_id;

    struct {
        xccl_mpod_nccl_req_s nccl[XCCL_MPOD_MAX_NCCL_PHASES];
        xccl_tl_coll_req_t *ucx_slice;
        xccl_tl_coll_req_t *ucx_flat;
    } real_req;

    struct {
        void *tmpbuf;
        xccl_mpod_buf_s *copied_hash;
        xccl_mc_event_t *event;
    } alltoall;
} xccl_mpod_chunk_s;

typedef struct xccl_mpod_coll_req {
    xccl_tl_coll_req_t super;

    xccl_mpod_team_t *team;
    ucs_memory_type_t memtype;
    xccl_coll_op_args_t coll_args;

    xccl_status_t (*collective_post)(struct xccl_mpod_coll_req *req);
    xccl_status_t (*collective_test)(struct xccl_mpod_coll_req *req);
    xccl_status_t (*collective_finalize)(struct xccl_mpod_coll_req *req);

    int num_chunks;
    xccl_mpod_chunk_s *chunks;

    int req_id;
    struct xccl_mpod_coll_req *self;
    UT_hash_handle hh;
} xccl_mpod_coll_req_t;

extern xccl_team_lib_mpod_t xccl_team_lib_mpod;

xccl_status_t xccl_mpod_nccl_req_init(xccl_mpod_coll_req_t *mpod_req,
                                      xccl_coll_op_args_t *coll_args,
                                      xccl_mpod_nccl_req_s *nccl_req);
xccl_status_t xccl_mpod_nccl_req_finalize(xccl_mpod_nccl_req_s *nccl_req);
xccl_status_t xccl_mpod_nccl_req_post(xccl_mpod_nccl_req_s *nccl_req);
xccl_status_t xccl_mpod_nccl_req_wait(xccl_mpod_nccl_req_s *nccl_req);
xccl_status_t xccl_mpod_nccl_req_test(xccl_mpod_nccl_req_s *nccl_req);

xccl_status_t xccl_mpod_barrier_init(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_cpu_init(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_allreduce_init(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_allreduce_init_split(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_allreduce_init_coalesce(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_allreduce_init_replicate(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_allgather_init(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_bcast_init(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_alltoall_init(xccl_mpod_coll_req_t *req);
xccl_status_t xccl_mpod_alltoallv_init(xccl_mpod_coll_req_t *req);

#endif  /* XCCL_TEAM_LIB_MPOD_H_ */
