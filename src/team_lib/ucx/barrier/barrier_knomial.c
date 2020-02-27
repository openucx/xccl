#include "config.h"
#include "tccl_ucx_lib.h"
#include "barrier.h"
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
        iteration   = req->barrier.iteration;         \
        radix_pow   = req->barrier.radix_mask_pow;    \
        active_reqs = req->barrier.active_reqs;       \
    }while(0)

#define SAVE_STATE(_phase) do{                         \
        req->barrier.phase          = _phase;        \
        req->barrier.iteration      = iteration;     \
        req->barrier.radix_mask_pow = radix_pow;     \
        req->barrier.active_reqs    = active_reqs;   \
    }while(0)

tccl_status_t
tccl_ucx_barrier_knomial_progress(tccl_ucx_collreq_t *req)
{
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, active_reqs, k, step_size, peer;
    tccl_team_h team   = req->team;
    int myrank        = team->oob.rank;
    int group_size    = team->oob.size;
    int radix         = req->barrier.radix;
    tccl_ucx_request_t **reqs = req->barrier.reqs;
    /* fprintf(stderr, "AR, radix %d, data_size %zd, count %d\n",
 radix, data_size, args->barrier.count); */
    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);
    RESTORE_STATE();
    GOTO_PHASE(req->barrier.phase);

    if (KN_EXTRA == node_type) {
            peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
            tccl_ucx_send_nb(NULL, 0, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[0]);
            tccl_ucx_recv_nb(NULL, 0, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[1]);
            active_reqs = 2;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        tccl_ucx_recv_nb(NULL, 0, peer,
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
        }
    }

    for (; iteration < pow_k_sup; iteration++) {
        step_size = radix_pow * radix;
        active_reqs = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            tccl_ucx_send_nb(NULL, 0, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }

        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            tccl_ucx_recv_nb(NULL, 0, peer,
                            (tccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }
        radix_pow *= radix;
        if (active_reqs) {
        PHASE_1:
            if (TCCL_INPROGRESS == tccl_ucx_testall((tccl_ucx_team_t *)team,
                                                       reqs, active_reqs)) {
                SAVE_STATE(PHASE_1);
                return TCCL_OK;
            }
        }
    }
    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        tccl_ucx_send_nb(NULL, 0, peer,
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
    return TCCL_OK;
}

tccl_status_t tccl_ucx_barrier_knomial_start(tccl_ucx_collreq_t *req)
{
    size_t data_size     = req->args.buffer_info.len;
    req->barrier.radix = 4; //TODO
    if (req->barrier.radix > req->team->oob.size) {
        req->barrier.radix = req->team->oob.size;
    }

    memset(req->barrier.reqs, 0, sizeof(req->barrier.reqs));
    req->barrier.phase          = PHASE_0;
    req->barrier.iteration      = 0;
    req->barrier.radix_mask_pow = 1;
    req->barrier.active_reqs    = 0;
    req->progress = tccl_ucx_barrier_knomial_progress;
    return tccl_ucx_barrier_knomial_progress(req);
}
