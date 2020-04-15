/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include <xccl_lib.h>
#include "xccl_team_lib.h"
#include "xccl_global_opts.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

extern xccl_lib_t xccl_static_lib;
extern xccl_config_t xccl_lib_global_config;
extern ucs_config_field_t xccl_lib_global_config_table[];

xccl_status_t xccl_team_lib_finalize(xccl_team_lib_h lib)
{
    dlclose(lib->dl_handle);
    return XCCL_OK;
}

__attribute__((destructor))
static void xccl_destructor(void)
{
    int i;

    ucs_config_parser_release_opts(&xccl_lib_global_config, xccl_lib_global_config_table);
    xccl_mem_component_finalize();

    for (i=0; i<xccl_static_lib.n_libs_opened; i++) {
        xccl_team_lib_finalize(xccl_static_lib.libs[i]);
    }
    if (xccl_static_lib.libs) {
        free(xccl_static_lib.libs);
    }
}
