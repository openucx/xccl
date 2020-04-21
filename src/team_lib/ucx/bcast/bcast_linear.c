#include "config.h"
#include "xccl_ucx_lib.h"
#include "bcast.h"
#include "xccl_ucx_sendrecv.h"
#include <stdlib.h>
#include <string.h>

xccl_status_t xccl_ucx_bcast_linear_progress(xccl_ucx_collreq_t *req)
{
    xccl_tl_team_t *team = req->team;
    void *data_buffer    = req->args.buffer_info.dst_buffer;
    size_t data_size     = req->args.buffer_info.len;
    int group_rank       = team->params.oob.rank;
    int group_size       = team->params.oob.size;
    xccl_ucx_request_t **reqs = req->bcast_linear.reqs;

    if (req->args.root == group_rank) {
        if (req->bcast_linear.step == ((group_rank + 1) % group_size)) {
            xccl_ucx_send_nb(data_buffer, data_size, req->bcast_linear.step,
                            (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->bcast_linear.step = ((req->bcast_linear.step + 1) % group_size);
        }
        if (XCCL_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs, 1)) {
            if (req->bcast_linear.step != group_rank) {
                xccl_ucx_send_nb(data_buffer, data_size, req->bcast_linear.step,
                                (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
                req->bcast_linear.step =
                    ((req->bcast_linear.step + 1) % group_size);
            } else {
                goto completion;
            }
        }

    } else {
        if (req->bcast_linear.step == 0) {
            xccl_ucx_recv_nb(data_buffer, data_size, req->args.root,
                            (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->bcast_linear.step = 1;
        }
        if (UCS_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs, 1)) {
            goto completion;
        }
    }
    return XCCL_OK;

completion:
    /* fprintf(stderr, "Complete bcast, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_bcast_linear_start(xccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    int group_rank   = req->team->params.oob.rank;
    int group_size   = req->team->params.oob.size;

    xccl_ucx_trace("linear bcast start");
    memset(req->bcast_linear.reqs, 0, sizeof(req->bcast_linear.reqs));
    req->bcast_linear.step    = 0;
    if (req->args.root == group_rank) {
        req->bcast_linear.step = (group_rank + 1) % group_size;
        if (req->args.buffer_info.src_buffer !=
            req->args.buffer_info.dst_buffer) {
            xccl_ucx_send_recv(req->args.buffer_info.src_buffer, data_size,
                               group_rank, req->tag, req->args.buffer_info.dst_buffer,
                               data_size, group_rank, req->tag,
                               (xccl_ucx_team_t *)req->team);
        }
    }
    req->progress = xccl_ucx_bcast_linear_progress;
    return xccl_ucx_bcast_linear_progress(req);
}
