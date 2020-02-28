/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TEAM_LIB_UCX_H_
#define XCCL_TEAM_LIB_UCX_H_
#include "xccl_team_lib.h"
#include <ucp/api/ucp.h>

typedef struct xccl_team_lib_ucx {
    xccl_team_lib_t super;
} xccl_team_lib_ucx_t;
extern xccl_team_lib_ucx_t xccl_team_lib_ucx;

typedef enum {
    XCCL_UCX_REQUEST_ACTIVE,
    XCCL_UCX_REQUEST_DONE,
} xccl_ucx_request_status_t;

typedef struct xccl_ucx_request_t {
    xccl_ucx_request_status_t status;
} xccl_ucx_request_t;

#define MAX_REQS 32

typedef struct xccl_ucx_collreq {
    xccl_coll_req_t     super;
    xccl_coll_op_args_t args;
    xccl_tl_team_t     *team;
    xccl_status_t       complete;
    uint16_t            tag;
    xccl_status_t       (*start)(struct xccl_ucx_collreq* req);
    xccl_status_t       (*progress)(struct xccl_ucx_collreq* req);
    union {
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            int                    phase;
            int                    iteration;
            int                    radix_mask_pow;
            int                    active_reqs;
            int                    radix;
            void                   *scratch;
        } allreduce;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                    step;
            void                   *scratch;
        } reduce_linear;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                    step;
        } fanin_linear;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                    step;
        } fanout_linear;
        struct {
            xccl_ucx_request_t *reqs[2];
            int                    step;
        } bcast_linear;
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            int                    active_reqs;
            int                    radix;
            int                    dist;
        } bcast_kn;
        struct {
            xccl_ucx_request_t *reqs[MAX_REQS];
            int                    phase;
            int                    iteration;
            int                    radix_mask_pow;
            int                    active_reqs;
            int                    radix;
        } barrier;
    };
} xccl_ucx_collreq_t;


#endif
