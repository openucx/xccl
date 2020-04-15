/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_SHARP_COLLECTIVE_H_
#define XCCL_TEAM_SHARP_COLLECTIVE_H_

#include "xccl_sharp_lib.h"

typedef struct xccl_sharp_coll_req {
    xccl_tl_coll_req_t            super;
    xccl_sharp_team_t             *team;
    struct sharp_coll_reduce_spec reduce_spec;
    struct sharp_coll_comm        *sharp_comm;
    void                          *handle;
    xccl_sharp_buf_t              *sharp_buf;
    xccl_collective_type_t        coll_type;
    int                           (*start)(struct xccl_sharp_coll_req* req);
    xccl_sharp_rcache_region_t    *src_rregion;
    xccl_sharp_rcache_region_t    *dst_rregion;
} xccl_sharp_coll_req_t;

xccl_status_t xccl_sharp_collective_init(xccl_coll_op_args_t *coll_args,
                                         xccl_coll_req_h *request,
                                         xccl_tl_team_t *team);
xccl_status_t xccl_sharp_collective_post(xccl_coll_req_h request);
xccl_status_t xccl_sharp_collective_wait(xccl_coll_req_h request);
xccl_status_t xccl_sharp_collective_test(xccl_coll_req_h request);
xccl_status_t xccl_sharp_collective_finalize(xccl_coll_req_h request);

#endif
