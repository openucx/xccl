/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include "tccl_team_lib.h"
#include <stdlib.h>
#include <dlfcn.h>



static tccl_status_t tccl_team_lib_finalize(tccl_team_lib_h lib) {
    dlclose(lib->dl_handle);
    return TCCL_OK;
}

tccl_status_t tccl_lib_finalize(tccl_lib_h lib)
{
    int i;
    for (i=0; i<lib->n_libs_opened; i++) {
        tccl_team_lib_finalize(lib->libs[i]);
    }
    if (lib->lib_path) {
        free(lib->lib_path);
    }
    if (lib->libs) {
        free(lib->libs);
    }
    return TCCL_OK;
}
