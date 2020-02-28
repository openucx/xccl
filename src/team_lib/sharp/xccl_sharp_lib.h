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
    xccl_team_lib_t super;
} xccl_team_lib_sharp_t;
extern xccl_team_lib_sharp_t xccl_team_lib_sharp;

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
