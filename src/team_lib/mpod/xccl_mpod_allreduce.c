#include "xccl_mpod_lib.h"

xccl_status_t xccl_mpod_allreduce_init(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    char *str = getenv("XCCL_MPOD_ALLREDUCE_ALGORITHM");
    if (str == NULL) {
        str = "replicate";
    }

    if (!strcmp(str, "replicate")) {
        status = xccl_mpod_allreduce_init_replicate(req);
        xccl_mpod_err_pop(status, fn_fail);
    } else if (!strcmp(str, "split")) {
        status = xccl_mpod_allreduce_init_split(req);
        xccl_mpod_err_pop(status, fn_fail);
    } else {
        status = xccl_mpod_allreduce_init_coalesce(req);
        xccl_mpod_err_pop(status, fn_fail);
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
