/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_GLOBAL_OPTS_H_
#define XCCL_GLOBAL_OPTS_H_

#include "config.h"

#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>

typedef struct xccl_config {
    /* Log level above which log messages will be printed*/
    ucs_log_component_config_t log_component;

    /* Team libraries path */
    char                       *team_lib_path;
} xccl_config_t;

#endif