/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "config.h"
#define _GNU_SOURCE
#include "xccl_team_lib.h"
#include <ucs/config/parser.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/compiler_def.h>
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
#include "utils/mem_component.h"
#include "utils/xccl_log.h"
#include "xccl_global_opts.h"
#include <xccl_lib.h>

xccl_lib_t xccl_static_lib;
xccl_local_proc_info_t xccl_local_proc_info;
extern xccl_config_t xccl_lib_global_config;
extern ucs_config_field_t xccl_lib_global_config_table[];

xccl_local_proc_info_t* xccl_local_process_info()
{
    return &xccl_local_proc_info;
}

static int
callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    if (NULL != (str = strstr(info->dlpi_name, "libxccl.so"))) {
        int pos = (int)(str - info->dlpi_name);
        free(xccl_lib_global_config.team_lib_path);
        xccl_lib_global_config.team_lib_path = (char*)malloc(pos+8);
        strncpy(xccl_lib_global_config.team_lib_path, info->dlpi_name, pos);
        xccl_lib_global_config.team_lib_path[pos] = '\0';
        strcat(xccl_lib_global_config.team_lib_path, "xccl");
    }
    return 0;
}

static void get_default_lib_path()
{
    dl_iterate_phdr(callback, NULL);
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
    xccl_debug("Loading library %s\n", so_path);
    if (!handle) {
        xccl_error("Failed to load XCCL Team library: %s\n. "
                   "Check XCCL_TEAM_LIB_PATH or LD_LIBRARY_PATH\n", so_path);
        *team_lib = NULL;
        return XCCL_ERR_NO_MESSAGE;
    }
    lib = (xccl_team_lib_t*)dlsym(handle, team_lib_struct);
    lib->dl_handle = handle;
    if (lib->team_lib_open != NULL) {
        xccl_team_lib_config_t *tl_config = malloc(lib->team_lib_config.size);
        ucs_config_parser_fill_opts(tl_config, lib->team_lib_config.table, "XCCL_", lib->team_lib_config.prefix, 0);
        lib->team_lib_open(lib, tl_config);
        ucs_config_parser_release_opts(tl_config, lib->team_lib_config.table);
        free(tl_config);
    }
    (*team_lib) = lib;
    return XCCL_OK;
}

static void load_team_lib_plugins(xccl_lib_t *lib)
{
    const char *tl_pattern = "/xccl_team_lib_*.so";
    glob_t globbuf;
    int i;
    char *pattern = (char*)malloc(strlen(xccl_lib_global_config.team_lib_path) +
                                  strlen(tl_pattern) + 1);

    strcpy(pattern, xccl_lib_global_config.team_lib_path);
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

__attribute__((constructor))
static void xccl_constructor(void)
{
    char *var;
    char hostname[256];
    ucs_status_t status;

    xccl_lib_t *lib = &xccl_static_lib;
    lib->libs = NULL;
    lib->n_libs_opened = 0;
    lib->libs_array_size = 0;

    status = ucs_config_parser_fill_opts(&xccl_lib_global_config, xccl_lib_global_config_table,
                                         "XCCL_", NULL, 1);
    
    if (strlen(xccl_lib_global_config.team_lib_path) == 0) {
        get_default_lib_path();
    }
    xccl_info("XCCL team lib path: %s", xccl_lib_global_config.team_lib_path);

    if (!xccl_lib_global_config.team_lib_path) {
        xccl_error("Failed to get xccl library path. set XCCL_TEAM_LIB_PATH.\n");
        return;
    }

    load_team_lib_plugins(lib);
    if (lib->n_libs_opened == 0) {
        xccl_error("XCCL init: couldn't find any xccl_team_lib_<name>.so plugins"
                " in %s\n", xccl_lib_global_config.team_lib_path);
        return;
    }

    xccl_mem_component_init(xccl_lib_global_config.team_lib_path);
    /* xccl_print_libs(&xccl_static_lib); */
    gethostname(hostname, sizeof(hostname));
    xccl_local_proc_info.node_hash = xccl_str_hash(hostname);
    xccl_get_bound_socket_id(&xccl_local_proc_info.socketid);
}

