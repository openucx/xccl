#include <xccl_global_opts.h>

xccl_config_t xccl_lib_global_config = {
    .log_component            = {UCS_LOG_LEVEL_WARN, "XCCL"},
    .team_lib_path            = "",
    .mem_component_cache_size = 4096
};

ucs_config_field_t xccl_lib_global_config_table[] = {
  {"LOG_LEVEL", "warn",
  "XCCL logging level. Messages with a level higher or equal to the selected "
  "will be printed.\n"
  "Possible values are: fatal, error, warn, info, debug, trace, data, func, poll.",
  ucs_offsetof(xccl_config_t, log_component),
  UCS_CONFIG_TYPE_LOG_COMP},

  {"TEAM_LIB_PATH", "",
  "Specifies team libraries location",
  ucs_offsetof(xccl_config_t, team_lib_path),
  UCS_CONFIG_TYPE_STRING},

  {"MEM_COMPONENT_CACHE_SIZE", "4096",
  "Size of memory component preallocated buffer size",
  ucs_offsetof(xccl_config_t, mem_component_cache_size),
  UCS_CONFIG_TYPE_MEMUNITS},

  NULL
};
UCS_CONFIG_REGISTER_TABLE(xccl_lib_global_config_table, "XCCL global", NULL,
                          xccl_lib_global_config)
