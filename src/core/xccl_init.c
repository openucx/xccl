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

xccl_lib_t xccl_static_lib;
xccl_local_proc_info_t xccl_local_proc_info;

xccl_lib_config_t xccl_lib_global_config = {
    .log_component = {UCS_LOG_LEVEL_WARN, "XCCL"},
    .team_lib_path = ""
};

ucs_config_field_t xccl_lib_global_config_table[] = {
  {"LOG_LEVEL", "warn",
  "XCCL logging level. Messages with a level higher or equal to the selected "
  "will be printed.\n"
  "Possible values are: fatal, error, warn, info, debug, trace, data, func, poll.",
  ucs_offsetof(xccl_lib_config_t, log_component),
  UCS_CONFIG_TYPE_LOG_COMP},

  {"TEAM_LIB_PATH", "",
  "Specifies team libraries location",
  ucs_offsetof(xccl_lib_config_t, team_lib_path),
  UCS_CONFIG_TYPE_STRING},

  NULL
};
UCS_CONFIG_REGISTER_TABLE(xccl_lib_global_config_table, "XCCL global", NULL,
                          xccl_lib_global_config)

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
    ucs_config_parser_warn_unused_env_vars_once("XCCL_");
    return xccl_create_context(lib, config, context_p);
}

static ucs_config_field_t xccl_config_table[] = {
  {"TEAMS", "all",
   "Comma-separated list of teams to use. The order is not meaningful.\n"
   " - all    : use all the avalable teams.\n"
   " - ucx    : team ucx"
   " - sharp  : team sharp"
   " - vmc    : team vmc"
   " - shmseg : team shmseg"
   " - hier   : hierarchical",
   ucs_offsetof(xccl_config_t, teams), UCS_CONFIG_TYPE_STRING_ARRAY},

   {NULL}
};
UCS_CONFIG_REGISTER_TABLE(xccl_config_table, "XCCL", NULL, xccl_config_t)

xccl_status_t xccl_config_read(const char *env_prefix, const char *filename,
                               xccl_config_t **config_p){
    xccl_config_t *config;
    xccl_status_t status;
    char full_prefix[128] = "XCCL_";

    config = malloc(sizeof(*config));
    if (config == NULL) {
        status = XCCL_ERR_NO_MEMORY;
        goto err;
    }

    if ((env_prefix != NULL) && (strlen(env_prefix) > 0)) {
        snprintf(full_prefix, sizeof(full_prefix), "%s%s", "XCCL_", env_prefix);
    }

    status = ucs_config_parser_fill_opts(config, xccl_config_table, full_prefix,
                                         NULL, 0);
    if (status != UCS_OK) {
        goto err_free;
    }

    *config_p = config;
    return XCCL_OK;

err_free:
    free(config);
err:
    return status;
}

void xccl_config_release(xccl_config_t *config)
{
    free(config);
}

void xccl_config_print(const xccl_config_t *config, FILE *stream,
                       const char *title, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, xccl_config_table, NULL,
                                 "XCCL_", print_flags);
}
