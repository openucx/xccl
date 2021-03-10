#include "xccl_mpod_lib.h"

#define INTER_POD_INITIATED  (0)
#define INTRA_POD_INITIATED  (1)

static xccl_status_t bcast_post(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    if (req->team->slice_id == req->coll_args.root % req->team->pod_size) {
        status = req->team->context->lib.ucx->collective_post(req->chunks[0].real_req.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);

        req->chunks[0].phase_id = INTER_POD_INITIATED;
    } else {
        status = xccl_mpod_nccl_req_post(&req->chunks[0].real_req.nccl[0]);
        xccl_mpod_err_pop(status, fn_fail);

        req->chunks[0].phase_id = INTRA_POD_INITIATED;
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t bcast_test(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    if (req->chunks[0].phase_id == INTER_POD_INITIATED) {
        status = req->team->context->lib.ucx->collective_test(req->chunks[0].real_req.ucx_slice);
        if (status == XCCL_INPROGRESS) {
            goto fn_exit;
        } else {
            xccl_mpod_err_pop(status, fn_fail);

            status = xccl_mpod_nccl_req_post(&req->chunks[0].real_req.nccl[0]);
            xccl_mpod_err_pop(status, fn_fail);

            req->chunks[0].phase_id = INTRA_POD_INITIATED;
        }
    }

    if (req->chunks[0].phase_id == INTRA_POD_INITIATED) {
        status = xccl_mpod_nccl_req_test(&req->chunks[0].real_req.nccl[0]);
        if (status == XCCL_INPROGRESS) {
            goto fn_exit;
        } else {
            xccl_mpod_err_pop(status, fn_fail);
        }
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t bcast_finalize(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    if (req->team->slice_id == req->coll_args.root % req->team->pod_size) {
        status = req->team->context->lib.ucx->collective_finalize(req->chunks[0].real_req.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);
    }

    status = xccl_mpod_nccl_req_finalize(&req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    free(req->chunks);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_bcast_init(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    req->chunks = (xccl_mpod_chunk_s *) malloc(sizeof(xccl_mpod_chunk_s));
    req->num_chunks = 1;

    if (req->team->slice_id == req->coll_args.root % req->team->pod_size) {
        xccl_coll_op_args_t ucx_coll_args = req->coll_args;
        ucx_coll_args.root = req->coll_args.root / req->team->pod_size;
        status = req->team->context->lib.ucx->collective_init(&ucx_coll_args, &req->chunks[0].real_req.ucx_slice,
                                                              req->team->team.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);
    }

    xccl_coll_op_args_t nccl_coll_args = req->coll_args;
    nccl_coll_args.root = req->coll_args.root % req->team->pod_size;
    status = xccl_mpod_nccl_req_init(req, &nccl_coll_args, &req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    req->collective_post = bcast_post;
    req->collective_test = bcast_test;
    req->collective_finalize = bcast_finalize;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
