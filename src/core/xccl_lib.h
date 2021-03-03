/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_LIB_H_
#define XCCL_LIB_H_

#include "config.h"
#include <api/xccl.h>

typedef struct xccl_lib_config {
    const char *tls;
} xccl_lib_config_t;

typedef struct xccl_team_lib xccl_team_lib_t;
typedef struct xccl_lib {
    int             n_libs_opened;
    int             libs_array_size;
    xccl_team_lib_t **libs;
} xccl_lib_t;

extern xccl_lib_t xccl_static_lib;

#endif
