#include "xccl_mpod_lib.h"

#define INTRA_POD_ALLGATHER_INITIATED  (0)
#define INTER_POD_ALLGATHER_INITIATED  (1)
#define INTRA_POD_BCAST_INITIATED  (2)

static xccl_status_t allgather_post(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = xccl_mpod_nccl_req_post(&req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    req->chunks[0].phase_id = INTRA_POD_ALLGATHER_INITIATED;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t allgather_test(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    switch (req->chunks[0].phase_id) {
    case INTRA_POD_ALLGATHER_INITIATED:
        status = xccl_mpod_nccl_req_test(&req->chunks[0].real_req.nccl[0]);
        if (status == XCCL_OK) {
            if (req->team->slice_id == 0) {
                status = req->team->context->lib.ucx->collective_post(req->chunks[0].real_req.ucx_slice);
                xccl_mpod_err_pop(status, fn_fail);
            }

            status = XCCL_INPROGRESS;
            req->chunks[0].phase_id = INTER_POD_ALLGATHER_INITIATED;
        }
        break;

    case INTER_POD_ALLGATHER_INITIATED:
        if (req->team->slice_id == 0) {
            status = req->team->context->lib.ucx->collective_test(req->chunks[0].real_req.ucx_slice);
        }
        if (status == XCCL_OK) {
            status = xccl_mpod_nccl_req_post(&req->chunks[0].real_req.nccl[1]);
            xccl_mpod_err_pop(status, fn_fail);

            status = XCCL_INPROGRESS;
            req->chunks[0].phase_id = INTRA_POD_BCAST_INITIATED;
        }
        break;

    case INTRA_POD_BCAST_INITIATED:
        status = xccl_mpod_nccl_req_test(&req->chunks[0].real_req.nccl[1]);
        break;
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t allgather_finalize(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = xccl_mpod_nccl_req_finalize(&req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    if (req->team->slice_id == 0) {
        status = req->team->context->lib.ucx->collective_finalize(req->chunks[0].real_req.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);
    }

    status = xccl_mpod_nccl_req_finalize(&req->chunks[0].real_req.nccl[1]);
    xccl_mpod_err_pop(status, fn_fail);

    free(req->chunks);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_allgather_init(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    req->chunks = (xccl_mpod_chunk_s *) malloc(sizeof(xccl_mpod_chunk_s));
    req->num_chunks = 1;

    /* phase 1 */
    xccl_coll_op_args_t nccl_coll_args = req->coll_args;
    nccl_coll_args.buffer_info.len /= req->team->num_pods;
    status = xccl_mpod_nccl_req_init(req, &nccl_coll_args, &req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    /* phase 2 */
    if (req->team->slice_id == 0) {
        xccl_coll_op_args_t ucx_coll_args = req->coll_args;
        ucx_coll_args.buffer_info.src_buffer = req->coll_args.buffer_info.dst_buffer;
        status = req->team->context->lib.ucx->collective_init(&ucx_coll_args, &req->chunks[0].real_req.ucx_slice,
                                                          req->team->team.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);
    }

    /* phase 3 */
    nccl_coll_args = req->coll_args;
    nccl_coll_args.coll_type = XCCL_BCAST;
    nccl_coll_args.root = 0;
    nccl_coll_args.buffer_info.src_buffer = nccl_coll_args.buffer_info.dst_buffer;
    status = xccl_mpod_nccl_req_init(req, &nccl_coll_args, &req->chunks[0].real_req.nccl[1]);
    xccl_mpod_err_pop(status, fn_fail);

    req->collective_post = allgather_post;
    req->collective_test = allgather_test;
    req->collective_finalize = allgather_finalize;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
