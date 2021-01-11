#include "config.h"
#include "xccl_ucx_lib.h"
#include "alltoall.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

static inline int get_peer(int rank, int size, int step)
{
    return (step - rank + size)%size;
}

xccl_status_t xccl_ucx_alltoall_linear_shift_progress(xccl_ucx_collreq_t *req)
{
    ptrdiff_t       sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t       rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team       = ucs_derived_of(req->team, xccl_ucx_team_t);
    int             max_polls   = TEAM_UCX_CTX(team)->num_to_probe;
    int             group_rank  = team->super.params.oob.rank;
    int             group_size  = team->super.params.oob.size;
    size_t          data_size   = req->args.buffer_info.len;
    int             n_polls     = 0;

    int           completed_idx, peer;
    void          *send_buf, *recv_buf;
    xccl_status_t status;

    while ((n_polls++ < max_polls) &&
           (req->alltoall_linear_shift.step < group_size)) {
        status = xccl_ucx_req_test(team, req->alltoall_linear_shift.reqs, 2,
                                   &completed_idx, 1, 2);
        if (status == XCCL_OK) {
            peer = get_peer(group_rank, group_size, req->alltoall_linear_shift.step);
            if (peer != group_rank) {
                xccl_ucx_send_recv((void*)(sbuf + peer*data_size), data_size,
                                   req->src_mem_type, group_rank, req->tag,
                                   req->alltoall_linear_shift.scratch, data_size,
                                   req->src_mem_type, group_rank, req->tag, team);
                xccl_ucx_send_nb(req->alltoall_linear_shift.scratch, data_size,
                                 req->src_mem_type, peer, team, req->tag,
                                 &req->alltoall_linear_shift.reqs[0]);
                xccl_ucx_recv_nb((void*)(rbuf + peer*data_size), data_size,
                                 req->dst_mem_type, peer, team, req->tag,
                                 &req->alltoall_linear_shift.reqs[1]);
            } else {
                if (sbuf != rbuf) {
                    xccl_ucx_send_recv((void*)(sbuf + peer*data_size), data_size,
                                       req->src_mem_type, group_rank, req->tag,
                                       (void*)(rbuf + peer*data_size), data_size,
                                       req->dst_mem_type, group_rank, req->tag,
                                       team);
                }
            }
            n_polls = 0;
            req->alltoall_linear_shift.step++;
        }
    }

    if (req->alltoall_linear_shift.step < group_size) {
        return XCCL_OK;
    }

    if (xccl_ucx_testall(team, req->alltoall_linear_shift.reqs, 2) == XCCL_INPROGRESS) {
        return XCCL_OK;
    }

    req->complete = XCCL_OK;
    if (req->alltoall_linear_shift.scratch) {
        xccl_mem_component_free(req->alltoall_linear_shift.scratch,
                                req->src_mem_type);
    }

    return XCCL_OK;
}

xccl_status_t xccl_ucx_alltoall_linear_shift_start(xccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    xccl_mem_component_alloc(&req->alltoall_linear_shift.scratch,
                             data_size, req->src_mem_type);

    req->alltoall_linear_shift.reqs[0] = NULL;
    req->alltoall_linear_shift.reqs[1] = NULL;
    req->alltoall_linear_shift.step    = 0;
    req->progress                      = xccl_ucx_alltoall_linear_shift_progress;

    return xccl_ucx_alltoall_linear_shift_progress(req);
}
