#include "config.h"
#include "xccl_ucx_lib.h"
#include "fanout.h"
#include "xccl_ucx_sendrecv.h"
#include <stdlib.h>
#include <string.h>

xccl_status_t xccl_ucx_fanout_linear_progress(xccl_ucx_collreq_t *req)
{
    xccl_tl_team_t *team      = req->team;
    int            group_rank = team->params.oob.rank;
    int            group_size = team->params.oob.size;
    xccl_ucx_request_t **reqs = req->fanout_linear.reqs;

    if (req->args.root == group_rank) {
        if (req->fanout_linear.step == ((group_rank + 1) % group_size)) {
            xccl_ucx_send_nb(NULL, 0, req->fanout_linear.step,
                            (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->fanout_linear.step = ((req->fanout_linear.step + 1) % group_size);
        }
        if (XCCL_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs, 1)) {
            if (req->fanout_linear.step != group_rank) {
                xccl_ucx_send_nb(NULL, 0, req->fanout_linear.step,
                                (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
                req->fanout_linear.step =
                    ((req->fanout_linear.step + 1) % group_size);
            } else {
                goto completion;
            }
        }

    } else {
        if (req->fanout_linear.step == 0) {
            xccl_ucx_recv_nb(NULL, 0, req->args.root,
                            (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->fanout_linear.step = 1;
        }
        if (UCS_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs, 1)) {
            goto completion;
        }
    }
    return XCCL_OK;

completion:
    /* fprintf(stderr, "Complete fanout, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_fanout_linear_start(xccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    int group_rank   = req->team->params.oob.rank;
    int group_size   = req->team->params.oob.size;
    memset(req->fanout_linear.reqs, 0, sizeof(req->fanout_linear.reqs));
    req->fanout_linear.step    = 0;
    if (req->args.root == group_rank) {
        req->fanout_linear.step = (group_rank + 1) % group_size;
    }
    req->progress = xccl_ucx_fanout_linear_progress;
    return xccl_ucx_fanout_linear_progress(req);
}
