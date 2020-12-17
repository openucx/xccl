#include "config.h"
#include "xccl_ucx_lib.h"
#include "allreduce.h"
#include "allreduce_knomial.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/reduce.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

#define RESTORE_STATE() do{                             \
        iteration   = req->allreduce.iteration;         \
        radix_pow   = req->allreduce.radix_mask_pow;    \
        active_reqs = req->allreduce.active_reqs;       \
    }while(0)

#define SAVE_STATE(_phase) do{                         \
        req->allreduce.phase          = _phase;        \
        req->allreduce.iteration      = iteration;     \
        req->allreduce.radix_mask_pow = radix_pow;     \
        req->allreduce.active_reqs    = active_reqs;   \
    }while(0)

xccl_status_t
xccl_ucx_allreduce_knomial_progress(xccl_ucx_collreq_t *req)
{
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, active_reqs, k, step_size, peer;
    ptrdiff_t recv_offset;
    void *dst_buffer;
    void *src_buffer;
    xccl_tl_team_t *team = req->team;
    size_t data_size     = req->args.buffer_info.len;
    int myrank           = team->params.oob.rank;
    int group_size       = team->params.oob.size;
    int radix            = req->allreduce.radix;
    void *scratch        = req->allreduce.scratch;
    xccl_ucx_request_t **reqs = req->allreduce.reqs;
    /* fprintf(stderr, "AR, radix %d, data_size %zd, count %d\n",
 radix, data_size, args->allreduce.count); */
    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    GOTO_PHASE(req->allreduce.phase);

    if (KN_EXTRA == node_type) {
        peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
        xccl_ucx_send_nb(req->args.buffer_info.src_buffer, data_size, peer,
                        (xccl_ucx_team_t *)team, req->tag, &reqs[0]);
        xccl_ucx_recv_nb(req->args.buffer_info.dst_buffer, data_size, peer,
                        (xccl_ucx_team_t *)team, req->tag, &reqs[1]);
        active_reqs = 2;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        xccl_ucx_recv_nb(scratch, data_size, peer,
                        (xccl_ucx_team_t *)team, req->tag, &reqs[0]);
        active_reqs = 1;
    }
PHASE_EXTRA:
    if (KN_PROXY == node_type || KN_EXTRA == node_type) {
        if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                   reqs, active_reqs)) {
            SAVE_STATE(PHASE_EXTRA);
            return XCCL_OK;
        }
        if (KN_EXTRA == node_type) {
            goto completion;
        } else {
            xccl_mem_component_reduce(req->args.buffer_info.src_buffer,
                                      scratch,
                                      req->args.buffer_info.dst_buffer,
                                      req->args.reduce_info.count,
                                      req->args.reduce_info.dt,
                                      req->args.reduce_info.op,
                                      req->src_mem_type);
        }
    }

    for (; iteration < pow_k_sup; iteration++) {
        src_buffer  = ((iteration == 0) && (node_type == KN_BASE)) ?
                      req->args.buffer_info.src_buffer:
                      req->args.buffer_info.dst_buffer;
        dst_buffer  = req->args.buffer_info.dst_buffer;
        step_size   = radix_pow * radix;
        active_reqs = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            xccl_ucx_send_nb(src_buffer, data_size, peer,
                            (xccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }

        recv_offset = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            xccl_ucx_recv_nb((void*)((ptrdiff_t)scratch + recv_offset), data_size,
                             peer, (xccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
            recv_offset += data_size;
        }
        radix_pow *= radix;
        if (active_reqs) {
        PHASE_1:
            src_buffer = ((iteration == 0) && (node_type == KN_BASE)) ?
                         req->args.buffer_info.src_buffer:
                         req->args.buffer_info.dst_buffer;
            dst_buffer = req->args.buffer_info.dst_buffer;
            if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                       reqs, active_reqs)) {
                SAVE_STATE(PHASE_1);
                return XCCL_OK;
            }
            assert(active_reqs % 2 == 0);

            xccl_mem_component_reduce_multi(src_buffer, scratch, dst_buffer,
                                            active_reqs/2,
                                            req->args.reduce_info.count,
                                            data_size,
                                            req->args.reduce_info.dt,
                                            req->args.reduce_info.op,
                                            req->src_mem_type);
        }
    }
    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        xccl_ucx_send_nb(req->args.buffer_info.dst_buffer, data_size, peer,
                        (xccl_ucx_team_t *)team, req->tag, &reqs[0]);
        active_reqs = 1;
        goto PHASE_PROXY;
    } else {
        goto completion;
    }

PHASE_PROXY:
    if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                               reqs, active_reqs)) {
        SAVE_STATE(PHASE_PROXY);
        return XCCL_OK;
    }

completion:
    /* fprintf(stderr, "Complete reduce, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = XCCL_OK;
    if (req->allreduce.scratch) {
        xccl_mem_component_free(req->allreduce.scratch, req->src_mem_type);
    }
    return XCCL_OK;
}

xccl_status_t xccl_ucx_allreduce_knomial_start(xccl_ucx_collreq_t *req)
{
    size_t data_size     = req->args.buffer_info.len;

    req->allreduce.radix = TEAM_UCX_CTX_REQ(req)->allreduce_kn_radix;
    if (req->allreduce.radix > req->team->params.oob.size) {
        req->allreduce.radix = req->team->params.oob.size;
    }

    memset(req->allreduce.reqs, 0, sizeof(req->allreduce.reqs));
    req->allreduce.phase          = PHASE_0;
    req->allreduce.iteration      = 0;
    req->allreduce.radix_mask_pow = 1;
    req->allreduce.active_reqs    = 0;
    req->progress                 = xccl_ucx_allreduce_knomial_progress;
    xccl_mem_component_alloc(&req->allreduce.scratch,
                             (req->allreduce.radix-1)*data_size,
                             req->src_mem_type);
    return xccl_ucx_allreduce_knomial_progress(req);
}
