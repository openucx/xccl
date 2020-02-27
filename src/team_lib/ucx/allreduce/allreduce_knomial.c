#include "config.h"
#include "tccl_ucx_lib.h"
#include "allreduce.h"
#include "tccl_ucx_sendrecv.h"
#include "utils/reduce.h"
#include <stdlib.h>
#include <string.h>

enum {
    KN_BASE,
    KN_PROXY,
    KN_EXTRA
};

#define CALC_POW_K_SUP(_size, _radix, _pow_k_sup, _full_tree_size) do{  \
        int pk = 1;                                                     \
        int fs = _radix;                                                \
        while (fs < _size) {                                            \
            pk++; fs*=_radix;                                           \
        }                                                               \
        _pow_k_sup = pk;                                                \
        _full_tree_size = (fs != _size) ? fs/_radix : fs;               \
    }while(0)

#define KN_RECURSIVE_SETUP(__radix, __myrank, __size, __pow_k_sup,      \
                           __full_tree_size, __n_full_subtrees,         \
                           __full_size, __node_type) do{                \
        CALC_POW_K_SUP(__size, __radix, __pow_k_sup, __full_tree_size); \
        __n_full_subtrees = __size / __full_tree_size;                  \
        __full_size = __n_full_subtrees*__full_tree_size;               \
        __node_type = __myrank >= __full_size ? KN_EXTRA :              \
            (__size > __full_size && __myrank < __size - __full_size ?  \
             KN_PROXY : KN_BASE);                                       \
    }while(0)

#define KN_RECURSIVE_GET_PROXY(__myrank, __full_size) (__myrank - __full_size)
#define KN_RECURSIVE_GET_EXTRA(__myrank, __full_size) (__myrank + __full_size)

enum {
    PHASE_0,
    PHASE_1,
    PHASE_EXTRA,
    PHASE_PROXY,
};

#define CHECK_PHASE(_p) case _p: goto _p; break;
#define GOTO_PHASE(_phase) do{                  \
        switch (_phase) {                       \
            CHECK_PHASE(PHASE_EXTRA);           \
            CHECK_PHASE(PHASE_PROXY);           \
            CHECK_PHASE(PHASE_1);               \
        case PHASE_0: break;                    \
        };                                      \
    } while(0)

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

tccl_status_t
tccl_ucx_allreduce_knomial_progress(tccl_ucx_collreq_t *req)
{
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, active_reqs, k, step_size, peer;
    ptrdiff_t recv_offset;
    tccl_tl_team_t *team = req->team;
    void *data_buffer    = req->args.buffer_info.dst_buffer;
    size_t data_size     =  req->args.buffer_info.len;
    int myrank           = team->oob.rank;
    int group_size       = team->oob.size;
    int radix            = req->allreduce.radix;
    void *scratch        = req->allreduce.scratch;
    tccl_ucx_request_t **reqs = req->allreduce.reqs;
    /* fprintf(stderr, "AR, radix %d, data_size %zd, count %d\n",
 radix, data_size, args->allreduce.count); */
    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    GOTO_PHASE(req->allreduce.phase);

    if (KN_EXTRA == node_type) {
            peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
            tccl_ucx_send_nb(req->args.buffer_info.src_buffer, data_size, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[0]);
            tccl_ucx_recv_nb(req->args.buffer_info.dst_buffer, data_size, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[1]);
            active_reqs = 2;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        tccl_ucx_recv_nb(scratch, data_size, peer,
                        (tccl_ucx_team_t *)team, req->tag, &reqs[0]);
        active_reqs = 1;
    }
PHASE_EXTRA:
    if (KN_PROXY == node_type || KN_EXTRA == node_type) {
        if (TCCL_INPROGRESS == tccl_ucx_testall((tccl_ucx_team_t *)team,
                                                   reqs, active_reqs)) {
            SAVE_STATE(PHASE_EXTRA);
            return TCCL_OK;
        }
        if (KN_EXTRA == node_type) {
            goto completion;
        } else {
            tccl_dt_reduce(data_buffer, scratch, data_buffer,
                          req->args.reduce_info.count,
                          req->args.reduce_info.dt,
                          req->args.reduce_info.op);
        }
    }

    for (; iteration < pow_k_sup; iteration++) {
        step_size = radix_pow * radix;
        active_reqs = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            tccl_ucx_send_nb(data_buffer, data_size, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }

        recv_offset = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            tccl_ucx_recv_nb((void*)((ptrdiff_t)scratch + recv_offset), data_size,
                            peer, (tccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
            recv_offset += data_size;
        }
        radix_pow *= radix;
        if (active_reqs) {
        PHASE_1:
            if (TCCL_INPROGRESS == tccl_ucx_testall((tccl_ucx_team_t *)team,
                                                       reqs, active_reqs)) {
                SAVE_STATE(PHASE_1);
                return TCCL_OK;
            }
            assert(active_reqs % 2 == 0);
            for (k=0; k<active_reqs/2; k++) {
                tccl_dt_reduce(data_buffer, (void*)((ptrdiff_t)scratch + k*data_size),
                              data_buffer, req->args.reduce_info.count,
                              req->args.reduce_info.dt,
                              req->args.reduce_info.op);
            }
        }
    }
    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        tccl_ucx_send_nb(data_buffer, data_size, peer,
                        (tccl_ucx_team_t *)team, req->tag, &reqs[0]);
        active_reqs = 1;
        goto PHASE_PROXY;
    } else {
        goto completion;
    }

PHASE_PROXY:
    if (TCCL_INPROGRESS == tccl_ucx_testall((tccl_ucx_team_t *)team,
                                               reqs, active_reqs)) {
        SAVE_STATE(PHASE_PROXY);
        return TCCL_OK;
    }

completion:
    /* fprintf(stderr, "Complete reduce, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = TCCL_OK;
    if (req->allreduce.scratch) {
        free(req->allreduce.scratch);
    }
    return TCCL_OK;
}

tccl_status_t tccl_ucx_allreduce_knomial_start(tccl_ucx_collreq_t *req)
{
    size_t data_size     = req->args.buffer_info.len;
    req->allreduce.radix = 4; //TODO
    if (req->allreduce.radix > req->team->oob.size) {
        req->allreduce.radix = req->team->oob.size;
    }

    memset(req->allreduce.reqs, 0, sizeof(req->allreduce.reqs));
    req->allreduce.phase          = PHASE_0;
    req->allreduce.iteration      = 0;
    req->allreduce.radix_mask_pow = 1;
    req->allreduce.active_reqs    = 0;
    req->allreduce.scratch        = malloc((req->allreduce.radix-1)*data_size);

    memcpy(req->args.buffer_info.dst_buffer,
           req->args.buffer_info.src_buffer,
           data_size);
    req->progress = tccl_ucx_allreduce_knomial_progress;
    return tccl_ucx_allreduce_knomial_progress(req);
}
