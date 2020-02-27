#include "config.h"
#include "tccl_ucx_lib.h"
#include "fanin.h"
#include "tccl_ucx_sendrecv.h"
#include <stdlib.h>
#include <string.h>

tccl_status_t tccl_ucx_fanin_linear_progress(tccl_ucx_collreq_t *req)
{
    tccl_team_h team = req->team;
    int group_rank   = team->oob.rank;
    int group_size   = team->oob.size;
    tccl_ucx_request_t **reqs = req->fanin_linear.reqs;
    if (req->args.root == group_rank) {
        if (req->fanin_linear.step == ((group_rank + 1) % group_size)) {
            tccl_ucx_recv_nb(NULL, 0, req->fanin_linear.step,
                            (tccl_ucx_team_t*)team, req->tag, &reqs[0]);
            req->fanin_linear.step = ((req->fanin_linear.step + 1) % group_size);
        }
        if (TCCL_OK == tccl_ucx_testall((tccl_ucx_team_t *)team, reqs, 1)) {
            if (req->fanin_linear.step != group_rank) {
                tccl_ucx_recv_nb(NULL, 0, req->fanin_linear.step,
                                (tccl_ucx_team_t*)team, req->tag, &reqs[0]);
                req->fanin_linear.step =
                    ((req->fanin_linear.step + 1) % group_size);
            } else {
                goto completion;
            }
        }
    } else {
        if (req->fanin_linear.step == 0) {
            tccl_ucx_send_nb(NULL, 0,
                            req->args.root, (tccl_ucx_team_t*)team,
                            req->tag, &reqs[0]);
            req->fanin_linear.step = 1;
        }
        if (TCCL_OK == tccl_ucx_testall((tccl_ucx_team_t *)team, reqs, 1)) {
            goto completion;
        }
    }
    return TCCL_OK;

completion:
    /* fprintf(stderr, "Complete fanin, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = TCCL_OK;
    return TCCL_OK;
}

tccl_status_t tccl_ucx_fanin_linear_start(tccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    int group_rank   = req->team->oob.rank;
    int group_size   = req->team->oob.size;
    memset(req->fanin_linear.reqs, 0, sizeof(req->fanin_linear.reqs));
    if (req->args.root == group_rank) {
        req->fanin_linear.step = (group_rank + 1) % group_size;
    } else {
        req->fanin_linear.step = 0;
    }
    req->progress = tccl_ucx_fanin_linear_progress;
    return tccl_ucx_fanin_linear_progress(req);
}
