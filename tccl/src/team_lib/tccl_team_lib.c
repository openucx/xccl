/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#define _GNU_SOURCE
#include "tccl_team_lib.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <fts.h>

static int
callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    tccl_lib_t *lib = (tccl_lib_t*)data;
    if (NULL != (str = strstr(info->dlpi_name, "libtccl.so"))) {
        int pos = (int)(str - info->dlpi_name);
        lib->lib_path = (char*)malloc(pos+8);
        strncpy(lib->lib_path, info->dlpi_name, pos);
        lib->lib_path[pos] = '\0';
        strcat(lib->lib_path, "tccl");
    }
    return 0;
}

static void get_default_lib_path(tccl_lib_t *lib)
{
    dl_iterate_phdr(callback, (void*)lib);
}

static tccl_status_t tccl_team_lib_init(const char *so_path,
                                        tccl_team_lib_h *team_lib)
{
    char team_lib_struct[128];
    void *handle;
    tccl_team_lib_t *lib;

    int pos = (int)(strstr(so_path, "tccl_team_lib_") - so_path);
    if (pos < 0) {
        return TCCL_ERR_NO_MESSAGE;
    }
    strncpy(team_lib_struct, so_path+pos, strlen(so_path) - pos - 3);
    team_lib_struct[strlen(so_path) - pos - 3] = '\0';
    handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Failed to load TCCL Team library: %s\n. "
                "Check TCCL_TEAM_LIB_PATH or LD_LIBRARY_PATH\n", so_path);
        *team_lib = NULL;
        return TCCL_ERR_NO_MESSAGE;
    }
    lib = (tccl_team_lib_t*)dlsym(handle, team_lib_struct);
    lib->dl_handle = handle;
    (*team_lib) = lib;
    return TCCL_OK;
}

static void load_team_lib_plugins(tccl_lib_t *lib)
{
    FTS *ftsp;
    FTSENT *p, *chp;
    int fts_options = FTS_COMFOLLOW | FTS_LOGICAL | FTS_NOCHDIR;
    char* const arr[2] = {lib->lib_path, NULL};

    if ((ftsp = fts_open(arr, fts_options, NULL)) == NULL) {
        warn("fts_open");
        return;
    }
    /* Initialize ftsp with as many argv[] parts as possible. */
    chp = fts_children(ftsp, 0);
    if (chp == NULL) {
        return;               /* no files to traverse */

    }
    while ((p = fts_read(ftsp)) != NULL) {
        switch (p->fts_info) {
        case FTS_D:
            /*directory should not be there, skip */
            break;
        case FTS_F:
            if (strstr(p->fts_name, "tccl_team_lib_") &&
                strstr(p->fts_name, ".so")) {
                /* printf("f %s\n", p->fts_name); */
                if (lib->n_libs_opened == lib->libs_array_size) {
                    lib->libs_array_size += 8;
                    lib->libs = (tccl_team_lib_t**)realloc(lib->libs,
                                                           lib->libs_array_size*sizeof(*lib->libs));
                }
                tccl_team_lib_init(p->fts_path, &lib->libs[lib->n_libs_opened]);
                lib->n_libs_opened++;
            }
            break;
        default:
            break;
        }
    }
    fts_close(ftsp);
}

static tccl_status_t tccl_team_lib_finalize(tccl_team_lib_h lib) {
    dlclose(lib->dl_handle);
    return TCCL_OK;
}

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{\
        if ((config.field_mask & TCCL_LIB_CONFIG_FIELD_ ## _CAP_FIELD) && \
            !(config. _cap & tl->config. _cap)) {                       \
        printf("Disqualifying team %s due to %s cap\n", tl->name, TCCL_PP_QUOTE(_CAP_FIELD));\
            tccl_team_lib_finalize(tl);                                 \
            lib->libs[i] = NULL;                                        \
            kept--;                                                     \
            continue;                                                   \
        }                                                               \
    } while(0)

static void tccl_lib_filter(tccl_lib_config_t config,
                            tccl_lib_t *lib) {
    int i;
    int kept = lib->n_libs_opened;
    for (i=0; i<lib->n_libs_opened; i++) {
        tccl_team_lib_t *tl = lib->libs[i];
        CHECK_LIB_CONFIG_CAP(reproducible, REPRODUCIBLE);
        CHECK_LIB_CONFIG_CAP(thread_mode,  THREAD_MODE);
        CHECK_LIB_CONFIG_CAP(team_usage,   TEAM_USAGE);
        CHECK_LIB_CONFIG_CAP(coll_types,   COLL_TYPES);
    }
    if (kept != lib->n_libs_opened) {
        tccl_team_lib_t **libs = (tccl_team_lib_t**)malloc(kept*sizeof(*libs));
        kept = 0;
        for (i=0; i<lib->n_libs_opened; i++) {
            if (lib->libs[i]) {
                libs[kept++] = lib->libs[i];
            }
        }
        free(lib->libs);
        lib->libs = libs;
        lib->n_libs_opened = kept;
    }
}

tccl_status_t tccl_lib_init(tccl_lib_config_t config,
                            tccl_lib_h *tccl_lib)
{
    char *var;
    tccl_lib_t *lib = (tccl_lib_t*)malloc(sizeof(*lib));
    lib->libs = NULL;
    lib->n_libs_opened = 0;
    lib->libs_array_size = 0;
    lib->lib_path = NULL;
    var = getenv("TCCL_TEAM_LIB_PATH");
    if (var) {
        lib->lib_path = strdup(var);
    } else {
        get_default_lib_path(lib);
    }
    if (!lib->lib_path) {
        fprintf(stderr, "Failed to get tccl library path. set TCCL_TEAM_LIB_PATH.\n");
        return TCCL_ERR_NO_MESSAGE;
    }
    /* printf("LIB PATH:%s\n", lib->lib_path); */
    load_team_lib_plugins(lib);
    if (lib->n_libs_opened == 0) {
        fprintf(stderr, "TCCL init: couldn't find any tccl_team_lib_<name>.so plugins.\n");
        return TCCL_ERR_NO_MESSAGE;
    }
    tccl_lib_filter(config, lib);
    (*tccl_lib) = lib;
    return TCCL_OK;
}

tccl_status_t tccl_team_lib_query(tccl_team_lib_h team_lib,
                                tccl_team_lib_attr_t *attr)
{
    if (attr->field_mask & TCCL_ATTR_FIELD_CONTEXT_CREATE_MODE) {
        attr->context_create_mode = team_lib->ctx_create_mode;
    }
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

static tccl_status_t tccl_create_team_context(tccl_team_lib_h lib,
                                              tccl_context_config_h config,
                                              tccl_context_h *team_ctx)
{
    tccl_status_t status;
    status = lib->create_team_context(lib, config, team_ctx);
    if (TCCL_OK == status) {
        (*team_ctx)->lib = lib;
        memcpy(&((*team_ctx)->cfg), config, sizeof(tccl_context_config_t));
    }
    return status;
}

static tccl_status_t find_tlib_by_name(tccl_lib_t *lib, const char *tlib_name,
                                       tccl_team_lib_t **tlib) {
    int i;
    for (i=0; i<lib->n_libs_opened; i++) {
        if (0 == strcmp(tlib_name, lib->libs[i]->name)) {
            *tlib = lib->libs[i];
            return TCCL_OK;
        }
    }
    *tlib = NULL;
    return TCCL_ERR_NO_ELEM;
}

tccl_status_t tccl_create_context(tccl_lib_h lib, tccl_context_config_t config,
                                  tccl_context_t **context)
{
    tccl_status_t status;
    if (config.field_mask & TCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME) {
        tccl_team_lib_t *tlib;
        if (TCCL_OK != (status = find_tlib_by_name(lib, config.team_lib_name, &tlib))) {
            return status;
        }
        return tccl_create_team_context(tlib, &config, context);
    } else {
        //TODO automatic team lib selection from the list of opened components with
        // prioritization
        return TCCL_ERR_NOT_IMPLEMENTED;
    }
    return TCCL_OK;
}

tccl_status_t tccl_destroy_context(tccl_context_h team_ctx)
{
    return team_ctx->lib->destroy_team_context(team_ctx);
}

tccl_status_t tccl_team_create_post(tccl_context_h team_ctx,
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

tccl_status_t tccl_context_progress(tccl_context_h team_ctx)
{
    if (team_ctx->lib->progress) {
        return team_ctx->lib->progress(team_ctx);
    }
    return TCCL_OK;
}
