#include "config.h"
#include "tccl_ucx_lib.h"
#include "reduce.h"
#include "tccl_ucx_sendrecv.h"
#include "utils/reduce.h"
#include <stdlib.h>
#include <string.h>

tccl_status_t tccl_ucx_reduce_linear_progress(tccl_ucx_collreq_t *req)
{
    tccl_team_h team = req->team;
    void *data_buffer = req->args.buffer_info.dst_buffer;
    size_t data_size =  req->args.buffer_info.len;
    int group_rank   = team->oob.rank;
    int group_size  = team->oob.size;
    void *scratch = req->reduce_linear.scratch;
    tccl_ucx_request_t **reqs = req->reduce_linear.reqs;
    if (req->args.root == group_rank) {
        if (req->reduce_linear.step == ((group_rank + 1) % group_size)) {
            tccl_ucx_recv_nb(scratch, data_size, req->reduce_linear.step,
                            (tccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->reduce_linear.step = ((req->reduce_linear.step + 1) % group_size);
        }
        if (TCCL_OK == tccl_ucx_testall((tccl_ucx_team_t *)team, reqs, 1)) {
            tccl_dt_reduce(scratch, data_buffer, data_buffer,
                          req->args.reduce_info.count,
                          req->args.reduce_info.dt,
                          req->args.reduce_info.op);

            if (req->reduce_linear.step != group_rank) {
                tccl_ucx_recv_nb(scratch, data_size, req->reduce_linear.step,
                                (tccl_ucx_team_t*)team, req->tag, &reqs[0]);
                req->reduce_linear.step =
                    ((req->reduce_linear.step + 1) % group_size);
            } else {
                goto completion;
            }
        }
    } else {
        if (req->reduce_linear.step == 0) {
            tccl_ucx_send_nb(req->args.buffer_info.src_buffer, data_size,
                            req->args.root, (tccl_ucx_team_t*)team,
                            req->tag, &reqs[0]);
            req->reduce_linear.step = 1;
        }
        if (TCCL_OK == tccl_ucx_testall((tccl_ucx_team_t *)team, reqs, 1)) {
            goto completion;
        }
    }
    return TCCL_OK;

completion:
    /* fprintf(stderr, "Complete reduce, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = TCCL_OK;
    if (req->reduce_linear.scratch) {
        free(req->reduce_linear.scratch);
    }
    return TCCL_OK;
}

tccl_status_t tccl_ucx_reduce_linear_start(tccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    int group_rank   = req->team->oob.rank;
    int group_size   = req->team->oob.size;
    memset(req->reduce_linear.reqs, 0, sizeof(req->reduce_linear.reqs));
    req->reduce_linear.step    = 0;
    if (req->args.root == group_rank) {
        req->reduce_linear.scratch = malloc(data_size);
        memcpy(req->args.buffer_info.dst_buffer,
               req->args.buffer_info.src_buffer,
               data_size);
        req->reduce_linear.step = (group_rank + 1) % group_size;
    } else {
        req->reduce_linear.scratch = NULL;
    }
    req->progress = tccl_ucx_reduce_linear_progress;
    return tccl_ucx_reduce_linear_progress(req);
}
