#include <xccl_team_lib.h>

ucs_config_field_t xccl_team_lib_config_table[] = {
  {"LOG_LEVEL", "warn",
  "XCCL logging level. Messages with a level higher or equal to the selected "
  "will be printed.\n"
  "Possible values are: fatal, error, warn, info, debug, trace, data, func, poll.",
  ucs_offsetof(xccl_team_lib_config_t, log_component),
  UCS_CONFIG_TYPE_LOG_COMP},

  {"PRIORITY", "-1",
  "XCCL team lib priority.\n"
  "Possible values are: [1,inf]",
  ucs_offsetof(xccl_team_lib_config_t, priority),
  UCS_CONFIG_TYPE_INT},

  {NULL}
};


ucs_config_field_t xccl_tl_context_config_table[] = {

  {NULL}
};
