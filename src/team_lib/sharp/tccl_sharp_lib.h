/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef TCCL_TEAM_LIB_SHARP_H_
#define TCCL_TEAM_LIB_SHARP_H_
#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>
#include "tccl_team_lib.h"

#define TCCL_SHARP_REG_BUF_SIZE 1024
#define TCCL_SHARP_REG_BUF_NUM  10

typedef struct tccl_team_lib_sharp {
    tccl_team_lib_t super;
} tccl_team_lib_sharp_t;
extern tccl_team_lib_sharp_t tccl_team_lib_sharp;

typedef struct tccl_sharp_context {
    tccl_tl_context_t             super;
    struct sharp_coll_context *sharp_context;
} tccl_sharp_context_t;

typedef struct tccl_sharp_buf {
    void *buf;
    void *mr;
    void *orig_src_buf;
    void *orig_dst_buf;
    int   used;
} tccl_sharp_buf_t;

typedef struct tccl_sharp_team {
    tccl_team_t             super;
    struct sharp_coll_comm *sharp_comm;
    tccl_sharp_buf_t        bufs[TCCL_SHARP_REG_BUF_NUM];
} tccl_sharp_team_t;
#endif
