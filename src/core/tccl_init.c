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
#include <glob.h>

tccl_lib_t tccl_static_lib;
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
    const char *tl_pattern = "/tccl_team_lib_*.so";
    glob_t globbuf;
    int i;
    char *pattern = (char*)malloc(strlen(lib->lib_path) + strlen(tl_pattern) + 1);

    strcpy(pattern, lib->lib_path);
    strcat(pattern, tl_pattern);
    glob(pattern, 0, NULL, &globbuf);
    free(pattern);
    for(i=0; i<globbuf.gl_pathc; i++) {
        if (lib->n_libs_opened == lib->libs_array_size) {
            lib->libs_array_size += 8;
            lib->libs = (tccl_team_lib_t**)realloc(lib->libs,
                                                   lib->libs_array_size*sizeof(*lib->libs));
        }
        tccl_team_lib_init(globbuf.gl_pathv[i], &lib->libs[lib->n_libs_opened]);
        lib->n_libs_opened++;
    }

    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
}

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{                      \
        if ((params->field_mask & TCCL_LIB_CONFIG_FIELD_ ## _CAP_FIELD) && \
            !(params-> _cap & tl->params. _cap)) {                       \
            printf("Disqualifying team %s due to %s cap\n",             \
                   tl->name, TCCL_PP_QUOTE(_CAP_FIELD));                \
            continue;                                                   \
        }                                                               \
    } while(0)

static void tccl_lib_filter(const tccl_params_t *params, tccl_lib_t *lib)
{
    int i;
    int n_libs = tccl_static_lib.n_libs_opened;
    lib->libs = (tccl_team_lib_t**)malloc(sizeof(tccl_team_lib_t*)*n_libs);
    lib->n_libs_opened = 0;
    for (i=0; i<n_libs; i++) {
        tccl_team_lib_t *tl = tccl_static_lib.libs[i];
        CHECK_LIB_CONFIG_CAP(reproducible, REPRODUCIBLE);
        CHECK_LIB_CONFIG_CAP(thread_mode,  THREAD_MODE);
        CHECK_LIB_CONFIG_CAP(team_usage,   TEAM_USAGE);
        CHECK_LIB_CONFIG_CAP(coll_types,   COLL_TYPES);
        
        lib->libs[lib->n_libs_opened++] = tl;
    }
}

static void tccl_print_libs(tccl_lib_t *lib) {
    char str[1024];
    int i;
    sprintf(str, "n_libs %d: ", lib->n_libs_opened);
    for (i=0; i<lib->n_libs_opened; i++) {
        strcat(str, lib->libs[i]->name);
        strcat(str, " ");
    }
    printf("%s\n", str);
}

__attribute__((constructor))
static void tccl_constructor(void)
{
    char *var;
    tccl_lib_t *lib = &tccl_static_lib;
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
        return;
    }
    /* printf("LIB PATH:%s\n", lib->lib_path); */
    load_team_lib_plugins(lib);
    if (lib->n_libs_opened == 0) {
        fprintf(stderr, "TCCL init: couldn't find any tccl_team_lib_<name>.so plugins.\n");
        return;
    }
    /* tccl_print_libs(&tccl_static_lib); */
}

tccl_status_t tccl_lib_init(const tccl_params_t *params,
                            tccl_lib_t **tccl_lib)
{
    tccl_lib_t *lib = NULL;
    if (tccl_static_lib.n_libs_opened == 0) {
        return TCCL_ERR_NO_MESSAGE;
    }

    lib = malloc(sizeof(*lib));
    lib->lib_path = NULL;
    tccl_lib_filter(params, lib);
    if (lib->n_libs_opened == 0) {
        fprintf(stderr, "TCCL lib init: no plugins left after filtering by params\n");
        return TCCL_ERR_NO_MESSAGE;
    }
    /* tccl_print_libs(lib); */
    *tccl_lib = lib;
    return TCCL_OK;
}

tccl_status_t tccl_init(const tccl_params_t *params,
                        const tccl_config_t *config,
                        tccl_context_h *context_p)
{
    tccl_status_t status;
    int i;
    tccl_lib_t *lib;

    if (TCCL_OK != (status = tccl_lib_init(params, &lib))) {
        return status;
    }
    return tccl_create_context(lib, config, context_p);
}
