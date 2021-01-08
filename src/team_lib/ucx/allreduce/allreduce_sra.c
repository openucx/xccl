#include "allreduce.h"
#include "allreduce_knomial.h"
#include "xccl_ucx_lib.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"

enum {
    SRA_KNOMIAL_SCATTER_REDUCE_START,
    SRA_KNOMIAL_SCATTER_REDUCE_PROGRESS,
    SRA_KNOMIAL_ALLGATHER_START,
    SRA_KNOMIAL_ALLGATHER_PROGRESS,
};

static inline int get_peer(int rank, int k, int radix_pow, int step_size)
{
    return (rank + k*radix_pow)%step_size + (rank - rank % step_size);
}

static inline int compute_step_radix(int myrank, int step_size, int radix_pow,
                                     int full_size, int radix)
{
    int step_radix = 0;
    int k, peer;

    for(k = 1; k < radix; k++) {
        peer = get_peer(myrank, k, radix_pow, step_size);
        if (peer >= full_size) continue;
        step_radix += 1;
    }
    step_radix += 1;

    return step_radix;
}

// segment index in exchange group of tree
static inline int compute_seg_index(int peer, int kpow_num, int tree_order)
{
    int peer_base, peer_position, peer_base_rank, peer_index;

    peer_base      = peer / (kpow_num * tree_order);
    peer_base_rank = peer_base * kpow_num * tree_order ;
    peer_position  = peer_base_rank == 0 ? peer : peer % (peer_base_rank);
    peer_index     = peer_position / kpow_num ;

    return peer_index;
}

// segment size
static inline int compute_seg_size(int block_count, int radix, int si)
{
    return block_count/radix + (si < (block_count % radix) ? 1 : 0);
}

// segment offset in exhange group of tree
static inline int compute_seg_offset(int block_count, int radix, int si)
{
    return (block_count/radix)*si +
           ((si < (block_count % radix)) ? si :  (block_count % radix));
}

static inline int compute_block_count(int count, int radix, int rank, int step)
{
    int block_count = count;
    int i, my_si, my_seg_len;
    int k_pow = 1;

    for (i=0; i<step; i++) {
        my_si       = compute_seg_index(rank, k_pow, radix);
        my_seg_len  = compute_seg_size(block_count, radix, my_si);
        block_count = my_seg_len;
        k_pow      *= radix;
    }
    return block_count;
}

void get_sra_knomial_offset_and_seglen(int count, size_t dt_size, int myrank,
                                       int radix, int group_size, ptrdiff_t *offset,
                                       int *seglen)
{
    ptrdiff_t _offset = 0;
    int block_count  = count;
    int radix_pow    = 1;
    int pow_k_sup;
    int full_tree_size;
    int full_size;
    int node_type;
    int n_full_subtrees;
    int k;
    int step_size;
    int r, step, peer, step_radix, my_si, my_seg_len;
    size_t my_seg_offset;

    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);

    if (KN_EXTRA == node_type) {
        if (offset) *offset = 0;
        if (seglen) *seglen = count;
        return;
    }
    for (step=0; step < pow_k_sup; step++) {
        r = 0;
        step_size   = radix_pow * radix;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size +
                   (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            r++;

        }
        step_radix    = r + 1;
        my_si         = compute_seg_index(myrank, radix_pow, radix);
        my_seg_offset = compute_seg_offset(block_count, step_radix, my_si);
        _offset += my_seg_offset*dt_size;
        if (step < pow_k_sup-1) {
            block_count = compute_seg_size(block_count, step_radix, my_si);
            radix_pow      *= radix;
        }
    }
    my_seg_len    = compute_seg_size(block_count, step_radix, my_si);
    if (offset) *offset = _offset;
    if (seglen) *seglen = my_seg_len;
}

ptrdiff_t get_sra_knomial_offset(int count, size_t dt_size, int myrank,
                                 int radix, int group_size)
{
    ptrdiff_t offset;
    get_sra_knomial_offset_and_seglen(count, dt_size, myrank, radix, group_size,
                                      &offset, NULL);
    return offset;
}

#define RESTORE_STATE() do{                              \
        iteration   = req->allreduce_sra.iteration;      \
        radix_pow   = req->allreduce_sra.radix_mask_pow; \
        active_reqs = req->allreduce_sra.active_reqs;    \
        step_radix  = req->allreduce_sra.step_radix;     \
    }while(0)

#define SAVE_STATE(_phase) do{                             \
        req->allreduce_sra.phase          = _phase;        \
        req->allreduce_sra.iteration      = iteration;     \
        req->allreduce_sra.radix_mask_pow = radix_pow;     \
        req->allreduce_sra.active_reqs    = active_reqs;   \
        req->allreduce_sra.step_radix     = step_radix;    \
    }while(0)

xccl_status_t xccl_ucx_scatter_reduce_knomial_progress(xccl_ucx_collreq_t *req)
{
    xccl_tl_team_t *team      = req->team;
    size_t data_size          = req->args.buffer_info.len;
    xccl_ucx_request_t **reqs = req->allreduce_sra.reqs;
    int radix                 = req->allreduce_sra.radix;
    int myrank                = team->params.oob.rank;
    int group_size            = team->params.oob.size;
    int dt_size               = xccl_dt_size(req->args.reduce_info.dt);
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, active_reqs, k, step_size, peer;
    int block_count, step_radix;
    int peer_seg_index, peer_seg_count, peer_seg_offset;
    int local_seg_index, local_seg_count, local_seg_offset;
    void *dst_buffer, *src_buffer;
    ucs_memory_type_t mtype;
    ptrdiff_t offset;
    int ret;

    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    block_count = compute_block_count(req->args.reduce_info.count,
                                      radix, myrank, iteration);
    GOTO_PHASE(req->allreduce_sra.phase);

    if ((req->args.buffer_info.src_buffer == req->args.buffer_info.dst_buffer) ||
        (KN_PROXY == node_type)) {
        ret = xccl_mem_component_alloc(&req->allreduce_sra.scratch,
                                       data_size, req->src_mem_type);
        if (ret != XCCL_OK) {
            xccl_ucx_error("Failed to to allocate %s memory. size:%ld",
                            ucs_memory_type_names[req->src_mem_type], data_size);
            return ret;
        }
    }

    if (KN_EXTRA == node_type) {
        peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
        xccl_ucx_send_nb(req->args.buffer_info.src_buffer, data_size,
                         req->src_mem_type, peer, (xccl_ucx_team_t*)team,
                         req->tag, &reqs[0]);
        active_reqs = 1;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        xccl_ucx_recv_nb(req->allreduce_sra.scratch, data_size, req->src_mem_type,
                         peer, (xccl_ucx_team_t*)team, req->tag, &reqs[0]);
        active_reqs = 1;
    }

PHASE_EXTRA:
    if ((KN_PROXY == node_type) || (KN_EXTRA == node_type)) {
        if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t*)team,
                                                reqs, active_reqs)) {
            SAVE_STATE(PHASE_EXTRA);
            return XCCL_INPROGRESS;
        }

        if (KN_EXTRA == node_type) {
            goto completion;
        } else {
            xccl_mem_component_reduce(req->args.buffer_info.src_buffer,
                                      req->allreduce_sra.scratch,
                                      req->args.buffer_info.dst_buffer,
                                      req->args.reduce_info.count,
                                      req->args.reduce_info.dt,
                                      req->args.reduce_info.op,
                                      req->src_mem_type);
            req->args.buffer_info.src_buffer = req->args.buffer_info.dst_buffer;
        }
    }

    for (; iteration < pow_k_sup; iteration++) {
        step_size   = radix_pow * radix;
        step_radix  = compute_step_radix(myrank, step_size, radix_pow, full_size, radix);
        block_count = compute_block_count(req->args.reduce_info.count,
                                          radix, myrank, iteration);
        src_buffer = (iteration == 0) ? req->args.buffer_info.src_buffer:
                     req->allreduce_sra.scratch;
        active_reqs = 0;
        for (k=1; k < radix; k++) {
            peer = get_peer(myrank, k, radix_pow, step_size);
            if (peer >= full_size) continue;

            peer_seg_index  = compute_seg_index(peer, radix_pow, radix);
            peer_seg_count  = compute_seg_size(block_count, step_radix, peer_seg_index);
            peer_seg_offset = compute_seg_offset(block_count, step_radix, peer_seg_index);
            xccl_ucx_send_nb((void*)((ptrdiff_t)src_buffer + (size_t)peer_seg_offset*dt_size),
                             peer_seg_count*dt_size, req->src_mem_type, peer,
                             (xccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }

        local_seg_index  = compute_seg_index(myrank, radix_pow, radix);
        local_seg_count  = compute_seg_size(block_count, step_radix, local_seg_index);

        dst_buffer = req->allreduce_sra.scratch;
        mtype = req->src_mem_type;
        if (iteration != 0) {
            dst_buffer = (void*)((ptrdiff_t)dst_buffer + (size_t)block_count*dt_size);
            mtype = req->dst_mem_type;
        }
        for (k = 1; k < radix; k++) {
            peer = get_peer(myrank, k, radix_pow, step_size);
            if (peer >= full_size) continue;

            xccl_ucx_recv_nb(dst_buffer, local_seg_count*dt_size, mtype,
                             peer, (xccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
            dst_buffer = (void*)((ptrdiff_t)dst_buffer + (size_t)(local_seg_count*dt_size));
        }
        if (active_reqs) {
PHASE_1:
            if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                    reqs, active_reqs)) {
                SAVE_STATE(PHASE_1);
                return XCCL_INPROGRESS;
            }
            src_buffer = req->args.buffer_info.src_buffer;
            dst_buffer = req->allreduce_sra.scratch;
            if (iteration != 0) {
                src_buffer = req->allreduce_sra.scratch;
                dst_buffer = (void*)((ptrdiff_t)dst_buffer + (size_t)block_count*dt_size);
            }
            assert(active_reqs%2 == 0);
            local_seg_index   = compute_seg_index(myrank, radix_pow, radix);
            local_seg_count   = compute_seg_size(block_count, step_radix, local_seg_index);
            local_seg_offset  = compute_seg_offset(block_count, step_radix, local_seg_index);
            void *local_data  = (void*)((ptrdiff_t)src_buffer + (size_t)local_seg_offset*dt_size);
            void *reduce_data = req->allreduce_sra.scratch;
            xccl_mem_component_reduce_multi(local_data, dst_buffer, reduce_data,
                                            active_reqs/2, local_seg_count,
                                            local_seg_count*dt_size, req->args.reduce_info.dt,
                                            req->args.reduce_info.op, req->src_mem_type);
            radix_pow *= radix;
        }
    }

    offset = get_sra_knomial_offset(req->args.reduce_info.count, dt_size, myrank,
                                    radix, group_size);
    xccl_ucx_send_recv(req->allreduce_sra.scratch, local_seg_count*dt_size,
                       req->src_mem_type, myrank, req->tag,
                       (void*)((ptrdiff_t)req->args.buffer_info.dst_buffer + offset),
                       data_size, req->dst_mem_type, myrank, req->tag,
                       (xccl_ucx_team_t *)req->team);
//prepare for allgather
    if ((req->args.buffer_info.src_buffer == req->args.buffer_info.dst_buffer) ||
        (KN_PROXY == node_type)) {
        xccl_mem_component_free(req->allreduce_sra.scratch, req->src_mem_type);
    }
    req->args.buffer_info.src_buffer = (void*)((ptrdiff_t)req->args.buffer_info.dst_buffer + offset);
    req->allreduce_sra.iteration = pow_k_sup - 1;
    req->allreduce_sra.radix_mask_pow = radix_pow / radix;
PHASE_PROXY:
completion:

    return XCCL_OK;
}


xccl_status_t xccl_ucx_scatter_reduce_knomial_start(xccl_ucx_collreq_t *req)
{
    memset(req->allreduce_sra.reqs, 0, sizeof(req->allreduce_sra.reqs));

    req->allreduce_sra.phase          = PHASE_0;
    req->allreduce_sra.iteration      = 0;
    req->allreduce_sra.radix_mask_pow = 1;
    req->allreduce_sra.active_reqs    = 0;
    req->allreduce_sra.scratch        = req->args.buffer_info.dst_buffer;

    return XCCL_OK;
}

xccl_status_t xccl_ucx_allgather_knomial_progress(xccl_ucx_collreq_t *req)
{
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, active_reqs, k, step_size, peer;
    ptrdiff_t recv_offset;
    void *dst_buffer;
    void *src_buffer;
    ucs_memory_type_t mtype = req->dst_mem_type;
    xccl_tl_team_t *team    = req->team;
    size_t data_size        = req->args.buffer_info.len;
    int myrank              = team->params.oob.rank;
    int group_size          = team->params.oob.size;
    int radix               = req->allreduce_sra.radix;
    void *scratch           = req->allreduce_sra.scratch;
    xccl_ucx_request_t **reqs = req->allreduce_sra.reqs;
    int dt_size = xccl_dt_size(req->args.reduce_info.dt);
    int block_count;
    int step_radix;

    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    block_count = compute_block_count(req->args.reduce_info.count,
                                      radix, myrank, iteration);
    GOTO_PHASE(req->allreduce_sra.phase);

    if (KN_EXTRA == node_type) {
        peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
        xccl_ucx_recv_nb(req->args.buffer_info.dst_buffer, data_size,
                         mtype, peer, (xccl_ucx_team_t *)team,
                         req->tag, &reqs[0]);
        active_reqs = 1;
    }
PHASE_EXTRA:
    if (KN_EXTRA == node_type) {
        if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                   reqs, active_reqs)) {
            SAVE_STATE(PHASE_EXTRA);
            return XCCL_INPROGRESS;
        }
        goto completion;
    }

    for (; iteration >= 0; iteration--) {
        int peer_seg_index, peer_seg_count, peer_seg_offset;
        int local_seg_index, local_seg_offset;

        active_reqs      = 0;
        step_size        = radix_pow * radix;
        block_count      = compute_block_count(req->args.reduce_info.count,
                                               radix, myrank, iteration);
        step_radix       = compute_step_radix(myrank, step_size, radix_pow, full_size, radix);
        local_seg_index  = compute_seg_index(myrank, radix_pow, radix);
        local_seg_offset = compute_seg_offset(block_count, step_radix, local_seg_index);
        src_buffer       = req->args.buffer_info.src_buffer;
        dst_buffer       = (void*)((ptrdiff_t)src_buffer - (size_t)local_seg_offset*dt_size);

        for (k=1; k < radix; k++) {
            peer = get_peer(myrank, k, radix_pow, step_size);
            if (peer >= full_size) continue;

            peer_seg_index  = compute_seg_index(myrank, radix_pow, radix);
            peer_seg_count  = compute_seg_size(block_count, step_radix, peer_seg_index);
            xccl_ucx_send_nb(src_buffer, peer_seg_count*dt_size, mtype, peer,
                             (xccl_ucx_team_t*)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }
        req->args.buffer_info.src_buffer = dst_buffer;

        for (k = 1; k < radix; k++) {
            peer = get_peer(myrank, k, radix_pow, step_size);
            if (peer >= full_size) continue;

            peer_seg_index  = compute_seg_index(peer, radix_pow, radix);
            peer_seg_count  = compute_seg_size(block_count, step_radix, peer_seg_index);
            peer_seg_offset = compute_seg_offset(block_count, step_radix, peer_seg_index);
            xccl_ucx_recv_nb((void*)((ptrdiff_t)dst_buffer + (size_t)peer_seg_offset*dt_size),
                             peer_seg_count * dt_size, mtype, peer,
                             (xccl_ucx_team_t*)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }
        if (active_reqs) {
PHASE_1:
            if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                       reqs, active_reqs)) {
                SAVE_STATE(PHASE_1);
                return XCCL_INPROGRESS;
            }
            radix_pow /= radix;
        }
    }
    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        xccl_ucx_send_nb(req->args.buffer_info.dst_buffer, data_size, mtype, peer,
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
        return XCCL_INPROGRESS;
    }

completion:
    return XCCL_OK;
}

xccl_status_t xccl_ucx_allgather_knomial_start(xccl_ucx_collreq_t *req)
{
    memset(req->allreduce_sra.reqs, 0, sizeof(req->allreduce_sra.reqs));

    req->allreduce_sra.phase          = PHASE_0;
    req->allreduce_sra.active_reqs    = 0;
    req->allreduce_sra.scratch        = req->args.buffer_info.dst_buffer;

    return XCCL_OK;
}

xccl_status_t xccl_ucx_allreduce_sra_progress(xccl_ucx_collreq_t *req)
{
    xccl_status_t st;
    int           repeat;

    do {
        repeat = 0;
        switch(req->allreduce_sra.allreduce_stage) {
        case SRA_KNOMIAL_SCATTER_REDUCE_START:
            st = xccl_ucx_scatter_reduce_knomial_start(req);
            if (st == XCCL_OK) {
                req->allreduce_sra.allreduce_stage = SRA_KNOMIAL_SCATTER_REDUCE_PROGRESS;
                repeat = 1;
            }
            break;
        case SRA_KNOMIAL_SCATTER_REDUCE_PROGRESS:
            st = xccl_ucx_scatter_reduce_knomial_progress(req);
            if (st == XCCL_OK) {
                req->allreduce_sra.allreduce_stage = SRA_KNOMIAL_ALLGATHER_START;
                repeat = 1;
            }
            break;
        case SRA_KNOMIAL_ALLGATHER_START:
            st = xccl_ucx_allgather_knomial_start(req);
            if (st == XCCL_OK) {
                req->allreduce_sra.allreduce_stage = SRA_KNOMIAL_ALLGATHER_PROGRESS;
                repeat = 1;
            }
            break;
        case SRA_KNOMIAL_ALLGATHER_PROGRESS:
            st = xccl_ucx_allgather_knomial_progress(req);
            if (st == XCCL_OK) {
                req->complete = XCCL_OK;
            }
        }
    } while(repeat);

    return XCCL_OK;
}

xccl_status_t xccl_ucx_allreduce_sra_start(xccl_ucx_collreq_t *req)
{
    int radix = TEAM_UCX_CTX_REQ(req)->allreduce_kn_radix;
    int count = req->args.reduce_info.count;

    if (radix > req->team->params.oob.size) {
        radix = req->team->params.oob.size;
    }

    /* if dst buffer is too small for scatter reduce use radix 2 */
    if (((count + radix - 1)/radix*(radix-1) > count) ||
        ((radix - 1) > count)) {
        radix = 2;
    }

    req->allreduce_sra.radix           = radix;
    req->allreduce_sra.allreduce_stage = SRA_KNOMIAL_SCATTER_REDUCE_START;
    req->progress                      = xccl_ucx_allreduce_sra_progress;
    req->complete                      = XCCL_INPROGRESS;

    return xccl_ucx_allreduce_sra_progress(req);
}
