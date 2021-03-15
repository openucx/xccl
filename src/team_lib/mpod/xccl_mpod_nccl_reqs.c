#include "xccl_mpod_lib.h"

xccl_status_t xccl_mpod_nccl_req_init(xccl_mpod_coll_req_t *mpod_req,
                                      xccl_coll_op_args_t *coll_args,
                                      xccl_mpod_nccl_req_s *nccl_req)
{
    xccl_status_t status = XCCL_OK;

    nccl_req->mpod_req = mpod_req;

    status = mpod_req->team->context->lib.nccl->collective_init(coll_args, &nccl_req->r, mpod_req->team->team.nccl);
    xccl_mpod_err_pop(status, fn_fail);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_nccl_req_finalize(xccl_mpod_nccl_req_s *nccl_req)
{
    xccl_status_t status = XCCL_OK;

    status = nccl_req->mpod_req->team->context->lib.nccl->collective_finalize(nccl_req->r);
    xccl_mpod_err_pop(status, fn_fail);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_nccl_req_post(xccl_mpod_nccl_req_s *nccl_req)
{
    xccl_status_t status = XCCL_OK;

    status = nccl_req->mpod_req->team->context->lib.nccl->collective_post(nccl_req->r);
    xccl_mpod_err_pop(status, fn_fail);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_nccl_req_wait(xccl_mpod_nccl_req_s *nccl_req)
{
    xccl_status_t status = XCCL_OK;

    do {
        status = xccl_mpod_nccl_req_test(nccl_req);
    } while (status == XCCL_INPROGRESS);

    return status;
}

xccl_status_t xccl_mpod_nccl_req_test(xccl_mpod_nccl_req_s *nccl_req)
{
    xccl_status_t status = XCCL_OK;

    status = nccl_req->mpod_req->team->context->lib.nccl->collective_test(nccl_req->r);
    xccl_mpod_err_pop(status, fn_fail);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
