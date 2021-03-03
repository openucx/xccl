#include "xccl_mpod_lib.h"
#include "mem_component.h"

#define INTER_POD_INITIATED  (0)
#define LOCAL_TRANSPOSE_1_INITIATED  (1)
#define INTRA_POD_INITIATED  (2)
#define LOCAL_TRANSPOSE_2_INITIATED  (3)

static void *get_buf_dst(void *src, xccl_mpod_coll_req_t *req, int num_rows, int num_cols)
{
    uintptr_t offset = ((char *) src - (char *) req->coll_args.buffer_info.dst_buffer) /
        req->coll_args.buffer_info.len;
    uintptr_t x = offset / num_cols;
    uintptr_t y = offset % num_cols;
    uintptr_t new_offset = y * num_rows + x;

    return (void *) ((char *) req->coll_args.buffer_info.dst_buffer + new_offset * req->coll_args.buffer_info.len);
}

static xccl_status_t transpose_dst_buf(xccl_mpod_coll_req_t *req, int num_rows, int num_cols)
{
    xccl_status_t status = XCCL_OK;
    xccl_mpod_buf_s *el, *tmp;

    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            if (i == j)
                continue;

            uintptr_t offset = (i * num_cols + j) * req->coll_args.buffer_info.len;
            void *buf = (void *) ((char *) req->coll_args.buffer_info.dst_buffer + offset);

            HASH_FIND_PTR(req->chunks[0].alltoall.copied_hash, &buf, el);

            if (el) {
                /* this buffer is a part of a cycle that has already been copied */
                continue;
            }

            /* keep a copy of the source data in the tmpbuf */
            status = xccl_mem_component_memcpy_async(buf, req->chunks[0].alltoall.tmpbuf,
                                                     req->coll_args.buffer_info.len,
                                                     &req->coll_args.stream);
            xccl_mpod_err_pop(status, fn_fail);

            /* push all coordinates connecting to our source buffer in
             * a hash till we find a cycle; once we find a cycle,
             * traverse back to shift data along this cycle. */
            void *b = buf;
            xccl_mpod_buf_s *hash = NULL;
            xccl_mpod_buf_s *last = NULL;
            while (1) {
                HASH_FIND_PTR(hash, &b, el);
                if (el) {
                    break;
                }

                el = (xccl_mpod_buf_s *) malloc(sizeof(xccl_mpod_buf_s));
                el->buf = b;
                HASH_ADD_PTR(hash, buf, el);
                last = el;

                b = get_buf_dst(b, req, num_rows, num_cols);
            }

            /* fire off a bunch of copies in the reverse order of
             * insertion, so we are guaranteed that the destination
             * buffer is always free when we want to copy into it. */
            for (el = last; el != NULL;) {
                tmp = el->hh.prev;

                void *dst = get_buf_dst(el->buf, req, num_rows, num_cols);
                if (el->buf != buf) {
                    status = xccl_mem_component_memcpy_async(el->buf, dst, req->coll_args.buffer_info.len,
                                                             &req->coll_args.stream);
                    xccl_mpod_err_pop(status, fn_fail);
                } else {
                    status = xccl_mem_component_memcpy_async(req->chunks[0].alltoall.tmpbuf, dst,
                                                             req->coll_args.buffer_info.len,
                                                             &req->coll_args.stream);
                    xccl_mpod_err_pop(status, fn_fail);
                }

                HASH_DEL(hash, el);
                HASH_ADD_PTR(req->chunks[0].alltoall.copied_hash, buf, el);

                el = tmp;
            }
        }
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t alltoall_post(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = req->team->context->lib.ucx->collective_post(req->chunks[0].real_req.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    req->chunks[0].phase_id = INTER_POD_INITIATED;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t alltoall_test(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    switch (req->chunks[0].phase_id) {
    case INTER_POD_INITIATED:
        status = req->team->context->lib.ucx->collective_test(req->chunks[0].real_req.ucx_slice);
        if (status == XCCL_OK) {
            status = transpose_dst_buf(req, req->team->num_pods, req->team->pod_size);
            xccl_mpod_err_pop(status, fn_fail);

            status = xccl_mc_event_record(&req->coll_args.stream, &req->chunks[0].alltoall.event);
            xccl_mpod_err_pop(status, fn_fail);

            status = XCCL_INPROGRESS;
            req->chunks[0].phase_id = LOCAL_TRANSPOSE_1_INITIATED;
        }
        break;

    case LOCAL_TRANSPOSE_1_INITIATED:
        status = xccl_mc_event_query(req->chunks[0].alltoall.event);
        if (status == XCCL_OK) {
            xccl_mc_event_free(req->chunks[0].alltoall.event);
            xccl_mpod_err_pop(status, fn_fail);

            status = xccl_mpod_nccl_req_post(&req->chunks[0].real_req.nccl[0]);
            xccl_mpod_err_pop(status, fn_fail);

            status = XCCL_INPROGRESS;
            req->chunks[0].phase_id = INTRA_POD_INITIATED;
        }
        break;

    case INTRA_POD_INITIATED:
        status = xccl_mpod_nccl_req_test(&req->chunks[0].real_req.nccl[0]);
        if (status == XCCL_OK) {
            status = transpose_dst_buf(req, req->team->pod_size, req->team->num_pods);
            xccl_mpod_err_pop(status, fn_fail);

            status = xccl_mc_event_record(&req->coll_args.stream, &req->chunks[0].alltoall.event);
            xccl_mpod_err_pop(status, fn_fail);

            status = XCCL_INPROGRESS;
            req->chunks[0].phase_id = LOCAL_TRANSPOSE_2_INITIATED;
        }
        break;

    case LOCAL_TRANSPOSE_2_INITIATED:
        status = xccl_mc_event_query(req->chunks[0].alltoall.event);
        if (status == XCCL_OK) {
            xccl_mc_event_free(req->chunks[0].alltoall.event);
            xccl_mpod_buf_s *el, *tmp;
            HASH_ITER(hh, req->chunks[0].alltoall.copied_hash, el, tmp) {
                HASH_DEL(req->chunks[0].alltoall.copied_hash, el);
                free(el);
            }
        }
        break;
    }

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

static xccl_status_t alltoall_finalize(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    status = xccl_mem_component_free(req->chunks[0].alltoall.tmpbuf, req->memtype);
    xccl_mpod_err_pop(status, fn_fail);

    status = req->team->context->lib.ucx->collective_finalize(req->chunks[0].real_req.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    status = xccl_mpod_nccl_req_finalize(&req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    free(req->chunks);

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}

xccl_status_t xccl_mpod_alltoall_init(xccl_mpod_coll_req_t *req)
{
    xccl_status_t status = XCCL_OK;

    req->chunks = (xccl_mpod_chunk_s *) malloc(sizeof(xccl_mpod_chunk_s));
    req->num_chunks = 1;

    req->chunks[0].alltoall.copied_hash = NULL;

    xccl_coll_op_args_t ucx_coll_args = req->coll_args;
    ucx_coll_args.buffer_info.len *= req->team->pod_size;
    status = req->team->context->lib.ucx->collective_init(&ucx_coll_args, &req->chunks[0].real_req.ucx_slice,
                                                          req->team->team.ucx_slice);
    xccl_mpod_err_pop(status, fn_fail);

    xccl_coll_op_args_t nccl_coll_args = req->coll_args;
    nccl_coll_args.buffer_info.len *= req->team->num_pods;
    status = xccl_mpod_nccl_req_init(req, &nccl_coll_args, &req->chunks[0].real_req.nccl[0]);
    xccl_mpod_err_pop(status, fn_fail);

    status = xccl_mem_component_alloc(&req->chunks[0].alltoall.tmpbuf,
                                      req->coll_args.buffer_info.len,
                                      req->memtype);
    xccl_mpod_err_pop(status, fn_fail);

    req->collective_post = alltoall_post;
    req->collective_test = alltoall_test;
    req->collective_finalize = alltoall_finalize;

  fn_exit:
    return status;
  fn_fail:
    goto fn_exit;
}
