#include <xccl_lib.h>
#include <ucs/config/types.h>
#include <ucs/debug/log_def.h>
#include <ucs/config/parser.h>
#include "utils/xccl_log.h"
#include "utils/mem_component.h"
#include <xccl_team_lib.h>

extern xccl_lib_t xccl_static_lib;

static ucs_config_field_t xccl_lib_config_table[] = {

   {NULL}
};
UCS_CONFIG_REGISTER_TABLE(xccl_lib_config_table, "XCCL", NULL, xccl_lib_config_t)

#define CHECK_LIB_CONFIG_CAP(_cap, _CAP_FIELD) do{                        \
        if ((params->field_mask & XCCL_LIB_PARAM_FIELD_ ## _CAP_FIELD) && \
            !(params-> _cap & tl->params. _cap)) {                        \
            xccl_info("Disqualifying team %s due to %s cap",              \
                      tl->name, UCS_PP_QUOTE(_CAP_FIELD));                \
            continue;                                                     \
        }                                                                 \
    } while(0)


static void xccl_lib_filter(const xccl_lib_params_t *params, xccl_lib_t *lib)
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

xccl_status_t xccl_lib_init(const xccl_lib_params_t *params,
                            const xccl_lib_config_t *config,
                            xccl_lib_h *xccl_lib)
{
    xccl_lib_t *lib;

    if (xccl_static_lib.n_libs_opened == 0) {
        return XCCL_ERR_NO_MESSAGE;
    }

    lib = malloc(sizeof(*lib));
    if (lib == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    xccl_lib_filter(params, lib);
    if (lib->n_libs_opened == 0) {
        xccl_error("XCCL lib init: no plugins left after filtering by params\n");
        return XCCL_ERR_NO_MESSAGE;
    }

    *xccl_lib = lib;
    //TODO: move to appropriate place
    //ucs_config_parser_warn_unused_env_vars_once("XCCL_");
    return XCCL_OK;
}

xccl_status_t xccl_lib_config_read(const char *env_prefix,
                                   const char *filename,
                                   xccl_lib_config_t **config_p){
    xccl_lib_config_t *config;
    xccl_status_t status;
    char full_prefix[128] = "XCCL_";

    config = malloc(sizeof(*config));
    if (config == NULL) {
        status = XCCL_ERR_NO_MEMORY;
        goto err;
    }

    if ((env_prefix != NULL) && (strlen(env_prefix) > 0)) {
        snprintf(full_prefix, sizeof(full_prefix), "%s_%s", env_prefix, "XCCL_");
    }

    status = ucs_config_parser_fill_opts(config, xccl_lib_config_table, full_prefix,
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

void xccl_lib_config_release(xccl_lib_config_t *config)
{
    free(config);
}

void xccl_lib_config_print(const xccl_lib_config_t *config, FILE *stream,
                           const char *title, ucs_config_print_flags_t print_flags)
{
    ucs_config_parser_print_opts(stream, title, config, xccl_lib_config_table,
                                 NULL, "XCCL_", print_flags);
}

void xccl_lib_cleanup(xccl_lib_h lib_p)
{
    if (lib_p->libs) {
        free(lib_p->libs);
    }
    xccl_mem_component_free_cache();
    free(lib_p);
}
