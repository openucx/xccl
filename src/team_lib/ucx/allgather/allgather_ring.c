#include "config.h"
#include "xccl_ucx_lib.h"
#include "allgather.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>


xccl_status_t xccl_ucx_allgather_ring_progress(xccl_ucx_collreq_t *req)
{
    int             group_rank = req->team->params.oob.rank;
    int             group_size = req->team->params.oob.size;
    int             data_size  = req->args.buffer_info.len / group_size;
    int             sendto     = (group_rank + 1) % group_size;
    int             recvfrom   = (group_rank - 1 + group_size) % group_size;
    xccl_ucx_team_t *team      = ucs_derived_of(req->team, xccl_ucx_team_t);
    int             max_polls  = TEAM_UCX_CTX(team)->num_to_probe;
    int             n_polls    = 0;
    int             step;
    ptrdiff_t       sbuf;
    ptrdiff_t       rbuf;
    xccl_status_t   status;
    int             cidx;

    while ((n_polls++ < max_polls) && (req->allgather_ring.step < group_size - 1)) {
        step = req->allgather_ring.step;
        if ((req->allgather_ring.reqs[0] == NULL) &&
            (req->allgather_ring.reqs[1] == NULL)) {
            sbuf = (ptrdiff_t)req->args.buffer_info.dst_buffer +
                   ((group_rank-step+group_size)%group_size)*data_size;
            xccl_ucx_send_nb((void*)sbuf, data_size, sendto, team, req->tag,
                             &req->allgather_ring.reqs[0]);
            rbuf = (ptrdiff_t)req->args.buffer_info.dst_buffer +
                   ((group_rank-step-1+group_size)%group_size)*data_size;
            xccl_ucx_recv_nb((void*)rbuf, data_size, recvfrom, team, req->tag,
                             &req->allgather_ring.reqs[1]);
        }
        status = xccl_ucx_req_test(team, req->allgather_ring.reqs, 2, &cidx, 1, 2);
        if (status == XCCL_OK) {
            req->allgather_ring.step += 1;
            n_polls = 0;
        }
    }

    if (req->allgather_ring.step < group_size - 1) {
        return XCCL_OK;
    }

    if (XCCL_INPROGRESS == xccl_ucx_testall(team, req->allgather_ring.reqs, 2)) {
        return XCCL_OK;
    }
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_allgather_ring_start(xccl_ucx_collreq_t *req)
{
    int       group_rank = req->team->params.oob.rank;
    int       group_size = req->team->params.oob.size;
    int       data_size  = req->args.buffer_info.len / group_size;
    ptrdiff_t sbuf       = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf       = (ptrdiff_t)req->args.buffer_info.dst_buffer;

    if (sbuf != rbuf) {
        xccl_ucx_send_recv((void*)(sbuf), data_size,
                            group_rank, req->tag,
                            (void*)(rbuf + data_size*group_rank), data_size,
                            group_rank, req->tag,
                            (xccl_ucx_team_t *)req->team);
    }
    req->allgather_ring.step     = 0;
    req->allgather_ring.reqs[0] = NULL;
    req->allgather_ring.reqs[1] = NULL;
    req->progress               = xccl_ucx_allgather_ring_progress;
    return xccl_ucx_allgather_ring_progress(req);
}
