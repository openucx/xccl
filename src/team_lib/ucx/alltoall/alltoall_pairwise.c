#include "config.h"
#include "xccl_ucx_lib.h"
#include "alltoall.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

static inline int get_recv_peer(int group_rank, int group_size,
                                int step, int is_reverse)
{
    if (is_reverse) {
        return (group_rank - 1 - step + group_size) % group_size;
    } else {
        return (group_rank + 1 + step) % group_size;
    }
}

static inline int get_send_peer(int group_rank, int group_size,
                                int step, int is_reverse)
{
    if (is_reverse) {
        return (group_rank + 1 + step) % group_size;
    } else {
        return (group_rank - 1 - step + group_size) % group_size;
    }
}

xccl_status_t xccl_ucx_alltoall_pairwise_progress(xccl_ucx_collreq_t *req)
{
    ptrdiff_t            sbuf = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t            rbuf = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t     *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    size_t          data_size = req->args.buffer_info.len;
    int            group_rank = team->super.params.oob.rank;
    int            group_size = team->super.params.oob.size;
    xccl_ucx_request_t **reqs = req->alltoall_pairwise.reqs;
    int                 chunk = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int               reverse = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int             max_polls = TEAM_UCX_CTX(team)->num_to_probe;    
    int            total_reqs = (chunk > group_size - 1 || chunk <= 0) ?
        group_size - 1 : chunk;
    int step, peer, released_slot, n_polls;
    xccl_status_t status;
    n_polls = 0;    
    while (n_polls++ < max_polls &&
           (req->alltoall_pairwise.n_rreqs != group_size - 1 ||
            req->alltoall_pairwise.n_sreqs != group_size - 1)) {
        if (req->alltoall_pairwise.n_rreqs < group_size - 1) {
            status = xccl_ucx_req_test(team, reqs, total_reqs, &released_slot, 1, 1);
            if (XCCL_OK == status) {
                peer = get_recv_peer(group_rank, group_size,
                                     req->alltoall_pairwise.n_rreqs, reverse);
                xccl_ucx_recv_nb((void*)(rbuf + peer * data_size),
                                 data_size, peer, team, req->tag, &reqs[released_slot]);
                req->alltoall_pairwise.n_rreqs++;
                n_polls = 0;
            }
        }
        if (req->alltoall_pairwise.n_sreqs < group_size - 1) {
            status = xccl_ucx_req_test(team, reqs+total_reqs, total_reqs,
                                       &released_slot, 1, 1);
            if (XCCL_OK == status) {
                peer = get_send_peer(group_rank, group_size,
                                     req->alltoall_pairwise.n_sreqs, reverse);
                xccl_ucx_send_nb((void*)(sbuf + peer * data_size),
                                 data_size, peer, team, req->tag,
                                 &reqs[released_slot+total_reqs]);
                req->alltoall_pairwise.n_sreqs++;
                n_polls = 0;
            }
        }
    }
    if (req->alltoall_pairwise.n_rreqs != group_size - 1 ||
        req->alltoall_pairwise.n_sreqs != group_size - 1) {
        return XCCL_OK;
    }
    
    if (XCCL_INPROGRESS == xccl_ucx_testall(team, reqs, 2*total_reqs)) {
        return XCCL_OK;
    }
    free(reqs);
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_alltoall_pairwise_start(xccl_ucx_collreq_t *req)
{
    size_t data_size      = req->args.buffer_info.len;
    int    group_rank     = req->team->params.oob.rank;
    int    group_size     = req->team->params.oob.size;
    int    chunk          = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_chunk;
    int    reverse        = TEAM_UCX_CTX_REQ(req)->alltoall_pairwise_reverse;
    int    total_reqs     = (chunk > group_size - 1 || chunk <= 0) ?
        group_size - 1 : chunk;
    ptrdiff_t sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    xccl_ucx_request_t **reqs;    
    int step, peer;
    req->alltoall_pairwise.reqs = calloc(total_reqs*2,
                                         sizeof(*req->alltoall_pairwise.reqs));
    if (!req->alltoall_pairwise.reqs) {
        return XCCL_ERR_NO_MEMORY;
    }
    reqs          = req->alltoall_pairwise.reqs;
    xccl_ucx_send_recv((void*)(sbuf+data_size*group_rank), data_size,
                       group_rank, req->tag, (void*)(rbuf+data_size*group_rank),
                       data_size, group_rank, req->tag,
                       (xccl_ucx_team_t *)req->team);
    for (step = 0; step < total_reqs; step++) {
        peer = get_recv_peer(group_rank, group_size, step, reverse);
        xccl_ucx_recv_nb((void*)(rbuf + peer * data_size),
                         data_size, peer, team, req->tag, &reqs[step]);
    }
    for (step = 0; step < total_reqs; step++) {
        peer = get_send_peer(group_rank, group_size, step, reverse);
        xccl_ucx_send_nb((void*)(sbuf + peer * data_size),
                         data_size, peer, team, req->tag, &reqs[step+total_reqs]);
    }
    req->alltoall_pairwise.n_sreqs = total_reqs;
    req->alltoall_pairwise.n_rreqs = total_reqs;

    req->progress = xccl_ucx_alltoall_pairwise_progress;
    return xccl_ucx_alltoall_pairwise_progress(req);
}
