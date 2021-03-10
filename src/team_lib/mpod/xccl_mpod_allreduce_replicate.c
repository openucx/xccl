#include "xccl_mpod_lib.h"

#define INTRA_POD_ALLREDUCE_INITIATED  (0)
#define INTER_POD_ALLREDUCE_QUEUED  (1)
#define INTER_POD_ALLREDUCE_INITIATED  (2)
#define INTER_POD_ALLREDUCE_COMPLETED  (5)

/* FIXME: the current default chunk size is very large.  This is
 * because UCX uses a single cache buffer for in-place allreduce.
 * With pipelining, we quickly run out of that buffer and have to
 * reallocate/reregister a new buffer.  This is expensive and only
 * useful for very large messages. */
#define DEFAULT_CHUNK_SIZE   (1024 * 1024 * 1024)

static xccl_status_t allreduce_post(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    for (int i = 0; i < req->num_chunks; i++) {
        status = xccl_mpod_nccl_req_post(&req->chunks[i].real_req.nccl[0]);
        xccl_mpod_err_pop(status, fn_fail);

        req->chunks[i].phase_id = INTRA_POD_ALLREDUCE_INITIATED;
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t allreduce_test(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    for (int i = 0; i < req->num_chunks; i++) {
        switch (req->chunks[i].phase_id) {
        case INTRA_POD_ALLREDUCE_INITIATED:
            status = xccl_mpod_nccl_req_test(&req->chunks[i].real_req.nccl[0]);
            if (status == XCCL_OK) {
                if (req->team->num_pods == 1) {
                    req->chunks[i].phase_id = INTER_POD_ALLREDUCE_COMPLETED;
                } else {
                    req->chunks[i].phase_id = INTER_POD_ALLREDUCE_QUEUED;
                }
            }
            break;

        case INTER_POD_ALLREDUCE_QUEUED:
            if (i == 0 || req->chunks[i-1].phase_id > req->chunks[i].phase_id) {
                status = req->team->context->lib.ucx->collective_post(req->chunks[i].real_req.ucx_slice);
                xccl_mpod_err_pop(status, fn_fail);

                req->chunks[i].phase_id = INTER_POD_ALLREDUCE_INITIATED;
            }
            break;

        case INTER_POD_ALLREDUCE_INITIATED:
            status = req->team->context->lib.ucx->collective_test(req->chunks[i].real_req.ucx_slice);
            if (status == XCCL_OK) {
                req->chunks[i].phase_id = INTER_POD_ALLREDUCE_COMPLETED;
            }
            break;

        case INTER_POD_ALLREDUCE_COMPLETED:
            break;
        }

        if (status != XCCL_OK && status != XCCL_INPROGRESS) {
            goto fn_fail;
        }
    }

    int num_completed_phases = 0;
    for (int i = 0; i < req->num_chunks; i++) {
        if (req->chunks[i].phase_id == INTER_POD_ALLREDUCE_COMPLETED) {
            num_completed_phases++;
        }
    }
    if (num_completed_phases != req->num_chunks) {
        status = XCCL_INPROGRESS;
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t allreduce_finalize(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    for (int i = 0; i < req->num_chunks; i++) {
        status = xccl_mpod_nccl_req_finalize(&req->chunks[i].real_req.nccl[0]);
        xccl_mpod_err_pop(status, fn_fail);

        status = req->team->context->lib.ucx->collective_finalize(req->chunks[i].real_req.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);
    }

    free(req->chunks);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_allreduce_init_replicate(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    int chunk_size;
    char *str = getenv("XCCL_MPOD_ALLREDUCE_CHUNK_SIZE");
    if (str) {
        chunk_size = atoi(str);
    } else {
        chunk_size = DEFAULT_CHUNK_SIZE;
    }

    if (chunk_size > req->coll_args.buffer_info.len) {
        chunk_size = req->coll_args.buffer_info.len;
    }

    req->num_chunks = req->coll_args.buffer_info.len / chunk_size;
    req->num_chunks += !!(req->coll_args.buffer_info.len % chunk_size);
    req->chunks = (xccl_mpod_chunk_s *) malloc(req->num_chunks * sizeof(xccl_mpod_chunk_s));

    /* We use an allreduce-allreduce-allgather algorithm here.  In the
     * first step, we perform an allreduce inside the POD using NCCL.
     * After this, each process in the POD performs an inter-pod
     * allreduce using UCX over a subset of the data.  Finally, we
     * perform an intra-pod allgather using NCCL to collect the data
     * from the other processes in the POD. */

    xccl_coll_op_args_t coll_args = req->coll_args;
    uintptr_t len_multiplier = req->coll_args.buffer_info.len / req->coll_args.reduce_info.count;
    for (int i = 0; i < req->num_chunks; i++) {
        if (i > 0) {
            coll_args.buffer_info.src_buffer =
                (void *) ((char *) coll_args.buffer_info.src_buffer + chunk_size);
            coll_args.buffer_info.dst_buffer =
                (void *) ((char *) coll_args.buffer_info.dst_buffer + chunk_size);
        }
        if (i < req->num_chunks - 1) {
            coll_args.buffer_info.len = chunk_size;
            coll_args.reduce_info.count = chunk_size / len_multiplier;
        } else {
            coll_args.buffer_info.len = req->coll_args.buffer_info.len -
                (req->num_chunks - 1) * chunk_size;
            coll_args.reduce_info.count = req->coll_args.reduce_info.count -
                (req->num_chunks - 1) * chunk_size / len_multiplier;
        }

        /* phase 1 */
        status = xccl_mpod_nccl_req_init(req, &coll_args, &req->chunks[i].real_req.nccl[0]);
        xccl_mpod_err_pop(status, fn_fail);


        /* phase 2 */
        xccl_coll_op_args_t ucx_coll_args = coll_args;
        ucx_coll_args.buffer_info.src_buffer = ucx_coll_args.buffer_info.dst_buffer;
        status = req->team->context->lib.ucx->collective_init(&ucx_coll_args, &req->chunks[i].real_req.ucx_slice,
                                                              req->team->team.ucx_slice);
        xccl_mpod_err_pop(status, fn_fail);
    }


    /* wrap up */
    req->collective_post = allreduce_post;
    req->collective_test = allreduce_test;
    req->collective_finalize = allreduce_finalize;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
