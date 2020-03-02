/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#define _GNU_SOURCE
#include "xccl_team_lib.h"
#include <ucs/debug/log.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <err.h>
#include <link.h>
#include <dlfcn.h>
#include <glob.h>
#include <utils/utils.h>
#include <unistd.h>

xccl_lib_t xccl_static_lib;
xccl_local_proc_info_t xccl_local_proc_info;

static int
callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    xccl_lib_t *lib = (xccl_lib_t*)data;
    if (NULL != (str = strstr(info->dlpi_name, "libxccl.so"))) {
        int pos = (int)(str - info->dlpi_name);
        lib->lib_path = (char*)malloc(pos+8);
        strncpy(lib->lib_path, info->dlpi_name, pos);
        lib->lib_path[pos] = '\0';
        strcat(lib->lib_path, "xccl");
    }
    return 0;
}

static void get_default_lib_path(xccl_lib_t *lib)
{
    dl_iterate_phdr(callback, (void*)lib);
}

static xccl_status_t xccl_team_lib_init(const char *so_path,
                                        xccl_team_lib_h *team_lib)
{
    char team_lib_struct[128];
    void *handle;
    xccl_team_lib_t *lib;

    int pos = (int)(strstr(so_path, "xccl_team_lib_") - so_path);
    if (pos < 0) {
        return XCCL_ERR_NO_MESSAGE;
    }
    strncpy(team_lib_struct, so_path+pos, strlen(so_path) - pos - 3);
    team_lib_struct[strlen(so_path) - pos - 3] = '\0';
    handle = dlopen(so_path, RTLD_LAZY);
    if (!handle) {
        xccl_error("Failed to load XCCL Team library: %s\n. "
                "Check XCCL_TEAM_LIB_PATH or LD_LIBRARY_PATH\n", so_path);
        *team_lib = NULL;
        return XCCL_ERR_NO_MESSAGE;
    }
    lib = (xccl_team_lib_t*)dlsym(handle, team_lib_struct);
    lib->dl_handle = handle;
    (*team_lib) = lib;
    return XCCL_OK;
}

static void load_team_lib_plugins(xccl_lib_t *lib)
{
    const char *tl_pattern = "/xccl_team_lib_*.so";
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
            lib->libs = (xccl_team_lib_t**)realloc(lib->libs,
                                                   lib->libs_array_size*sizeof(*lib->libs));
        }
        xccl_team_lib_init(globbuf.gl_pathv[i], &lib->libs[lib->n_libs_opened]);
        lib->n_libs_opened++;
    }

    if (globbuf.gl_pathc > 0) {
        globfree(&globbuf);
    }
}

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{                      \
        if ((params->field_mask & XCCL_LIB_CONFIG_FIELD_ ## _CAP_FIELD) && \
            !(params-> _cap & tl->params. _cap)) {                       \
            printf("Disqualifying team %s due to %s cap\n",             \
                   tl->name, XCCL_PP_QUOTE(_CAP_FIELD));                \
            continue;                                                   \
        }                                                               \
    } while(0)

static void xccl_lib_filter(const xccl_params_t *params, xccl_lib_t *lib)
{
    int i;
    int n_libs = xccl_static_lib.n_libs_opened;
    lib->libs = (xccl_team_lib_t**)malloc(sizeof(xccl_team_lib_t*)*n_libs);
    lib->n_libs_opened = 0;
    for (i=0; i<n_libs; i++) {
        xccl_team_lib_t *tl = xccl_static_lib.libs[i];
        CHECK_LIB_CONFIG_CAP(reproducible, REPRODUCIBLE);
        CHECK_LIB_CONFIG_CAP(thread_mode,  THREAD_MODE);
        CHECK_LIB_CONFIG_CAP(team_usage,   TEAM_USAGE);
        CHECK_LIB_CONFIG_CAP(coll_types,   COLL_TYPES);
        
        lib->libs[lib->n_libs_opened++] = tl;
    }
}

static void xccl_print_libs(xccl_lib_t *lib) {
    char str[1024];
    int i;
    sprintf(str, "n_libs %d: ", lib->n_libs_opened);
    for (i=0; i<lib->n_libs_opened; i++) {
        strcat(str, lib->libs[i]->name);
        strcat(str, " ");
    }
    printf("%s\n", str);
}

extern const char *ucs_log_level_names[];
static int __find_string_in_list(const char *str, const char **list)
{
    int i;

    for (i = 0; *list; ++list, ++i) {
        if (strcasecmp(*list, str) == 0) {
            return i;
        }
    }
    return -1;
}

__attribute__((constructor))
static void xccl_constructor(void)
{
    char *var;
    char hostname[256];
    xccl_lib_t *lib = &xccl_static_lib;
    lib->libs = NULL;
    lib->n_libs_opened = 0;
    lib->libs_array_size = 0;
    lib->lib_path = NULL;
    var = getenv("XCCL_LOG_LEVEL");
    if (var) {
        int level;
        level = __find_string_in_list(var, ucs_log_level_names);
        if (level < 0) {
            lib->log_config.log_level = UCS_LOG_LEVEL_TRACE;
        } else {
            lib->log_config.log_level = level;
        }
    } else {
        lib->log_config.log_level = UCS_LOG_LEVEL_WARN;
    }

    var = getenv("XCCL_TEAM_LIB_PATH");
    if (var) {
        lib->lib_path = strdup(var);
    } else {
        get_default_lib_path(lib);
    }
    xccl_info("XCCL team lib path: %s", lib->lib_path);
    if (!lib->lib_path) {
        xccl_error("Failed to get xccl library path. set XCCL_TEAM_LIB_PATH.\n");
        return;
    }
    /* printf("LIB PATH:%s\n", lib->lib_path); */
    load_team_lib_plugins(lib);
    if (lib->n_libs_opened == 0) {
        xccl_error("XCCL init: couldn't find any xccl_team_lib_<name>.so plugins.\n");
        return;
    }
    /* xccl_print_libs(&xccl_static_lib); */
    gethostname(hostname, sizeof(hostname));
    xccl_local_proc_info.node_hash = xccl_str_hash(hostname);
    xccl_get_bound_socket_id(&xccl_local_proc_info.socketid);
}

xccl_status_t xccl_lib_init(const xccl_params_t *params,
                            xccl_lib_t **xccl_lib)
{
    xccl_lib_t *lib = NULL;
    if (xccl_static_lib.n_libs_opened == 0) {
        return XCCL_ERR_NO_MESSAGE;
    }

    lib = malloc(sizeof(*lib));
    lib->lib_path = NULL;
    xccl_lib_filter(params, lib);
    if (lib->n_libs_opened == 0) {
        xccl_error("XCCL lib init: no plugins left after filtering by params\n");
        return XCCL_ERR_NO_MESSAGE;
    }
    /* xccl_print_libs(lib); */
    *xccl_lib = lib;
    return XCCL_OK;
}

xccl_status_t xccl_init(const xccl_params_t *params,
                        const xccl_config_t *config,
                        xccl_context_h *context_p)
{
    xccl_status_t status;
    int i;
    xccl_lib_t *lib;

    if (XCCL_OK != (status = xccl_lib_init(params, &lib))) {
        return status;
    }
    return xccl_create_context(lib, config, context_p);
}
