/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#include "tccl_team_lib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>

tccl_status_t tccl_team_lib_init(tccl_team_lib_params_t *tccl_params,
                               tccl_team_lib_h *team_lib)
{
    const int tccl_team_lib_name_max = 16;
    int len;
    char *var, *lib_path, *soname;
    char team_lib_struct[128];
    void *handle;
    tccl_team_lib_t *lib;
    
    var = getenv("TCCL_TEAM_LIB_PATH");
    if (var) {
        lib_path = var;
    } else {
        lib_path = "";
    }

    len = strlen(lib_path) + strlen("/tccl_team_lib_") +
        tccl_team_lib_name_max + strlen(".so") + 1;
    soname = (char*)malloc(len*sizeof(char));
    strcpy(soname, lib_path);
    if (0 != strcmp(lib_path, "")) {
        strcat(soname, "/");
    }
    strcat(soname, "tccl_team_lib_");
    strcat(soname, tccl_params->team_lib_name);
    strcat(soname, ".so");
    handle = dlopen(soname, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Failed to load TCCL Team library: %s\n. "
                "Check TCCL_TEAM_LIB_PATH or LD_LIBRARY_PATH\n", soname);
        free(soname);
        *team_lib = NULL;
        return TCCL_ERR_NO_MESSAGE;
    }
    free(soname);
    sprintf(team_lib_struct, "tccl_team_lib_%s", tccl_params->team_lib_name);
    lib = (tccl_team_lib_t*)dlsym(handle, team_lib_struct);
    lib->dl_handle = handle;
    memcpy(&lib->params, tccl_params, sizeof(*tccl_params));
    (*team_lib) = lib;
    return TCCL_OK;
}

tccl_status_t tccl_team_lib_finalize(tccl_team_lib_h lib) {
    dlclose(lib);
}

tccl_status_t tccl_team_lib_query(tccl_team_lib_h team_lib,
                                tccl_team_lib_attr_t *attr)
{
    if (attr->field_mask & TCCL_ATTR_FIELD_CONTEXT_CREATE_MODE) {
        attr->context_create_mode = team_lib->ctx_create_mode;
    }
    return TCCL_OK;
}

tccl_status_t tccl_create_team_context(tccl_team_lib_h lib,
                                     tccl_team_context_config_h config,
                                     tccl_team_context_h *team_ctx)
{
    tccl_status_t status;
    status = lib->create_team_context(lib, config, team_ctx);
    if (TCCL_OK == status) {
        (*team_ctx)->lib = lib;
        memcpy(&((*team_ctx)->cfg), config, sizeof(tccl_team_context_config_t));
    }
    return status;
}

tccl_status_t tccl_destroy_team_context(tccl_team_context_h team_ctx)
{
    return team_ctx->lib->destroy_team_context(team_ctx);
}

tccl_status_t tccl_team_create_post(tccl_team_context_h team_ctx,
                                  tccl_team_config_h config,
                                  tccl_oob_collectives_t oob, tccl_team_h *team)
{
    tccl_status_t status;
    status = team_ctx->lib->team_create_post(team_ctx, config, oob, team);
    if (TCCL_OK == status) {
        (*team)->ctx = team_ctx;
        memcpy(&((*team)->cfg), config, sizeof(tccl_team_config_t));
    }
    return status;
}

tccl_status_t tccl_team_destroy(tccl_team_h team)
{
    return team->ctx->lib->team_destroy(team);
}

tccl_status_t tccl_collective_init(tccl_coll_op_args_t *coll_args,
                                 tccl_coll_req_h *request, tccl_team_h team)
{
    return team->ctx->lib->collective_init(coll_args, request, team);
}

tccl_status_t tccl_collective_post(tccl_coll_req_h request)
{
    return request->lib->collective_post(request);
}

tccl_status_t tccl_collective_wait(tccl_coll_req_h request)
{
        return request->lib->collective_wait(request);
}

tccl_status_t tccl_collective_test(tccl_coll_req_h request)
{
        return request->lib->collective_test(request);
}

tccl_status_t tccl_collective_finalize(tccl_coll_req_h request)
{
    return request->lib->collective_finalize(request);
}

tccl_status_t tccl_context_progress(tccl_team_context_h team_ctx)
{
    if (team_ctx->lib->progress) {
        return team_ctx->lib->progress(team_ctx);
    }
    return TCCL_OK;
}
