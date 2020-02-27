/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include "tccl_team_lib.h"
#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>

extern tccl_lib_t tccl_static_lib;
tccl_status_t tccl_team_lib_finalize(tccl_team_lib_h lib)
{
    dlclose(lib->dl_handle);
    return TCCL_OK;
}

tccl_status_t tccl_lib_finalize(tccl_lib_h lib)
{
    int i;
    if (lib->libs) {
        free(lib->libs);
    }
    free(lib);
    return TCCL_OK;
}

__attribute__((destructor))
static void tccl_destructor(void)
{
    int i;
    for (i=0; i<tccl_static_lib.n_libs_opened; i++) {
        tccl_team_lib_finalize(tccl_static_lib.libs[i]);
    }
    if (tccl_static_lib.libs) {
        free(tccl_static_lib.libs);
    }
    if (tccl_static_lib.lib_path) {
        free(tccl_static_lib.lib_path);
    }
}
