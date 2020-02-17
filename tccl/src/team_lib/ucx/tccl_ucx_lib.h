/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_TEAM_LIB_UCX_H_
#define TCCL_TEAM_LIB_UCX_H_
#include "tccl_team_lib.h"
#include <ucp/api/ucp.h>

typedef struct tccl_team_lib_ucx {
    tccl_team_lib_t super;
} tccl_team_lib_ucx_t;
extern tccl_team_lib_ucx_t tccl_team_lib_ucx;

typedef enum {
    TCCL_UCX_REQUEST_ACTIVE,
    TCCL_UCX_REQUEST_DONE,
} tccl_ucx_request_status_t;

typedef struct tccl_ucx_request_t {
    tccl_ucx_request_status_t status;
} tccl_ucx_request_t;

#define MAX_REQS 32

typedef struct tccl_ucx_collreq {
    tccl_coll_req_t     super;
    tccl_coll_op_args_t args;
    tccl_team_h         team;
    tccl_status_t       complete;
    uint16_t           tag;
    tccl_status_t       (*start)(struct tccl_ucx_collreq* req);
    tccl_status_t       (*progress)(struct tccl_ucx_collreq* req);
    union {
        struct {
            tccl_ucx_request_t *reqs[MAX_REQS];
            int                    phase;
            int                    iteration;
            int                    radix_mask_pow;
            int                    active_reqs;
            int                    radix;
            void                   *scratch;
        } allreduce;
        struct {
            tccl_ucx_request_t *reqs[2];
            int                    step;
            void                   *scratch;
        } reduce_linear;
        struct {
            tccl_ucx_request_t *reqs[2];
            int                    step;
        } fanin_linear;
        struct {
            tccl_ucx_request_t *reqs[2];
            int                    step;
        } fanout_linear;
        struct {
            tccl_ucx_request_t *reqs[2];
            int                    step;
        } bcast_linear;
        struct {
            tccl_ucx_request_t *reqs[MAX_REQS];
            int                    active_reqs;
            int                    radix;
            int                    dist;
        } bcast_kn;
        struct {
            tccl_ucx_request_t *reqs[MAX_REQS];
            int                    phase;
            int                    iteration;
            int                    radix_mask_pow;
            int                    active_reqs;
            int                    radix;
        } barrier;
    };
} tccl_ucx_collreq_t;


#endif
