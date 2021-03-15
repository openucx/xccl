#include "xccl_mpod_lib.h"

static xccl_status_t cpu_post(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = req->team->context->lib.ucx->collective_finalize(req->chunks[0].real_req.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t cpu_test(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = req->team->context->lib.ucx->collective_test(req->chunks[0].real_req.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t cpu_finalize(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = req->team->context->lib.ucx->collective_finalize(req->chunks[0].real_req.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

    free(req->chunks);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_cpu_init(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    req->chunks = (xccl_mpod_chunk_s *) malloc(sizeof(xccl_mpod_chunk_s));
    req->num_chunks = 1;

    status = req->team->context->lib.ucx->collective_init(&req->coll_args, &req->chunks[0].real_req.ucx_flat,
                                                          req->team->team.ucx_flat);
    xccl_mpod_err_pop(status, fn_fail);

    req->collective_post = cpu_post;
    req->collective_test = cpu_test;
    req->collective_finalize = cpu_finalize;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
