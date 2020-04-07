#include "config.h"
#include "xccl_ucx_lib.h"
#include "reduce.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/reduce.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

xccl_status_t xccl_ucx_reduce_linear_progress(xccl_ucx_collreq_t *req)
{
    xccl_tl_team_t     *team        = req->team;
    void               *data_buffer = req->args.buffer_info.dst_buffer;
    size_t             data_size    = req->args.buffer_info.len;
    int                group_rank   = team->oob.rank;
    int                group_size   = team->oob.size;
    void               *scratch     = req->reduce_linear.scratch;
    xccl_ucx_request_t **reqs       = req->reduce_linear.reqs;

    if (req->args.root == group_rank) {
        if (req->reduce_linear.step == ((group_rank + 1) % group_size)) {
            xccl_ucx_recv_nb(scratch, data_size, req->reduce_linear.step,
                            (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->reduce_linear.step = ((req->reduce_linear.step + 1) % group_size);
        }
        if (XCCL_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs, 1)) {
            xccl_mem_component_reduce(scratch,
                                      data_buffer,
                                      data_buffer,
                                      req->args.reduce_info.count,
                                      req->args.reduce_info.dt,
                                      req->args.reduce_info.op,
                                      req->mem_type);

            if (req->reduce_linear.step != group_rank) {
                xccl_ucx_recv_nb(scratch, data_size, req->reduce_linear.step,
                                (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
                req->reduce_linear.step =
                    ((req->reduce_linear.step + 1) % group_size);
            } else {
                goto completion;
            }
        }
    } else {
        if (req->reduce_linear.step == 0) {
            xccl_ucx_send_nb(req->args.buffer_info.src_buffer, data_size,
                            req->args.root, (xccl_ucx_team_t*)team,
                            req->tag, &reqs[0]);
            req->reduce_linear.step = 1;
        }
        if (XCCL_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs, 1)) {
            goto completion;
        }
    }
    return XCCL_OK;

completion:
    /* fprintf(stderr, "Complete reduce, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = XCCL_OK;
    if (req->reduce_linear.scratch) {
        xccl_mem_component_free(req->reduce_linear.scratch, req->mem_type);
    }
    return XCCL_OK;
}

xccl_status_t xccl_ucx_reduce_linear_start(xccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    int group_rank   = req->team->oob.rank;
    int group_size   = req->team->oob.size;

    memset(req->reduce_linear.reqs, 0, sizeof(req->reduce_linear.reqs));
    req->reduce_linear.step    = 0;
    if (req->args.root == group_rank) {
        xccl_mem_component_alloc(&req->reduce_linear.scratch,
                                 data_size,
                                 req->mem_type);
        xccl_ucx_send_recv(req->args.buffer_info.src_buffer, data_size,
                           group_rank, req->tag, req->args.buffer_info.dst_buffer,
                           data_size, group_rank, req->tag,
                           (xccl_ucx_team_t *)req->team);
        req->reduce_linear.step = (group_rank + 1) % group_size;
    } else {
        req->reduce_linear.scratch = NULL;
    }
    req->progress = xccl_ucx_reduce_linear_progress;
    return xccl_ucx_reduce_linear_progress(req);
}
