/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef UCX_TEAM_SENDRECV_H_
#define UCX_TEAM_SENDRECV_H_
#include "xccl_ucx_tag.h"
#include "xccl_ucx_context.h"
#include "xccl_ucx_team.h"
#include <assert.h>

void xccl_ucx_send_completion_cb(void* request, ucs_status_t status);
void xccl_ucx_recv_completion_cb(void* request, ucs_status_t status,
                                     ucp_tag_recv_info_t *info);

#define TEAM_UCX_CTX(_team) (xccl_derived_of((_team)->super.ctx, xccl_team_lib_ucx_context_t))
#define TEAM_UCX_WORKER(_team) TEAM_UCX_CTX(_team)->ucp_worker

#define TEAM_UCX_MAKE_TAG(_tag, _rank, _context_id)                 \
    ((((uint64_t) (_tag))        << TEAM_UCX_TAG_BITS_OFFSET)  |    \
     (((uint64_t) (_rank))       << TEAM_UCX_RANK_BITS_OFFSET) |    \
     (((uint64_t) (_context_id)) << TEAM_UCX_CONTEXT_BITS_OFFSET))

#define TEAM_UCX_MAKE_SEND_TAG(_tag,  _rank, _context_id) \
    TEAM_UCX_MAKE_TAG(_tag, _rank, _context_id)

#define TEAM_UCX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag,           \
                               _src, _context) do {                     \
        assert((_tag)     <= TEAM_UCX_MAX_TAG);                         \
        assert((_src)     <= TEAM_UCX_MAX_RANK);                        \
        assert((_context) <= TEAM_UCX_MAX_CONTEXT);                     \
        (_ucp_tag_mask) = (uint64_t)(-1);                               \
        (_ucp_tag) = TEAM_UCX_MAKE_TAG((_tag), (_src), (_context));     \
    } while(0)


#define TEAM_UCX_CHECK_REQ_STATUS() do {                                \
        if (UCS_PTR_IS_ERR(ucx_req)) {                                  \
            fprintf(stderr,"Error in %s: tag %d; dest %d; worker_id"    \
                    " %d; errmsg %s\n",                                 \
                    __func__, tag, dest_group_rank,                     \
                    *((uint16_t *) &TEAM_UCX_WORKER(team)),             \
                    ucs_status_string(UCS_PTR_STATUS(ucx_req)));        \
            ucp_request_cancel(TEAM_UCX_WORKER(team), ucx_req);         \
            xccl_ucx_req_free(ucx_req);                             \
            *req = NULL;                                                \
            return UCS_ERR_NO_MESSAGE;                                  \
        }                                                               \
    } while(0)

#define TEAM_UCX_CHECK_SEND_REQ() do{           \
        TEAM_UCX_CHECK_REQ_STATUS();            \
        *req = ucx_req;                         \
    } while(0)

#define TEAM_UCX_CHECK_RECV_REQ() do {                          \
        TEAM_UCX_CHECK_REQ_STATUS();                            \
        /* Is this necessary? Can it call _cb? */               \
        ucp_tag_recv_info_t info;                               \
        ucs_status_t status = ucp_request_test(ucx_req, &info); \
        if (status == UCS_INPROGRESS) {                         \
            *req = ucx_req;                                     \
        } else {                                                \
            xccl_ucx_req_free(ucx_req);                     \
            *req = NULL;                                        \
        }                                                       \
    } while(0)


#define TEAM_UCX_WAIT_REQ(_req) do {                                    \
        if (!(_req)) {                                                  \
            return XCCL_SUCCESS;                                         \
        }                                                               \
        if (UCS_PTR_IS_ERR((_req))) {                                   \
            fprintf(stderr, "Error in %s;  dest %d;"                    \
                    " ep %d; errmsg %s",__FUNCTION__,                   \
                    dest_group_rank, *((uint16_t *)ep),                 \
                    ucs_status_string(UCS_PTR_STATUS((_req))));         \
            return XCCL_ERROR;                                           \
        }                                                               \
                                                                        \
        while (UCS_INPROGRESS == ucp_request_check_status((_req))) {    \
            ucp_worker_progress(TEAM_UCX_WORKER(team));                 \
        }                                                               \
        xccl_ucx_req_free((_req));                                  \
    } while(0)

static inline ucp_ep_h get_p2p_ep(xccl_ucx_team_t *team, int rank)
{
    ucp_ep_h ep;
    if (TEAM_UCX_CTX(team)->ucp_eps) {
        ep = TEAM_UCX_CTX(team)->ucp_eps[
            xccl_range_to_rank(team->super.cfg.range, rank)];
    } else {
        ep = team->ucp_eps[rank];
    }
    return ep;
}

static inline void xccl_ucx_req_free(xccl_ucx_request_t *req)
{
    req->status = XCCL_UCX_REQUEST_ACTIVE;
    ucp_request_free(req);
}

static inline xccl_status_t
xccl_ucx_send_nb(void *buffer, size_t msglen, int dest_group_rank,
                xccl_ucx_team_t *team, uint32_t tag, xccl_ucx_request_t **req)
{
    ucp_datatype_t datatype = ucp_dt_make_contig(msglen);
    ucp_tag_t ucp_tag       =
        TEAM_UCX_MAKE_SEND_TAG(tag, team->super.oob.rank, team->ctx_id);
    /* fprintf(stderr,"send to group_rank %d, len %d, tag %d\n", dest_group_rank, msglen, tag); */
    ucp_ep_h ep = get_p2p_ep(team, dest_group_rank);
    xccl_ucx_request_t *ucx_req = (xccl_ucx_request_t *)
        ucp_tag_send_nb(ep, buffer, 1, datatype,
                        ucp_tag, xccl_ucx_send_completion_cb);
    TEAM_UCX_CHECK_SEND_REQ();
    return XCCL_OK;
}

static inline xccl_status_t
xccl_ucx_recv_nb(void *buffer, size_t msglen, int dest_group_rank,
                xccl_ucx_team_t *team, uint32_t tag, xccl_ucx_request_t **req)
{
    ucp_datatype_t datatype = ucp_dt_make_contig(msglen);
    ucp_tag_t ucp_tag, ucp_tag_mask;
    /* fprintf(stderr,"recv from group_rank %d, len %d, tag %d\n", dest_group_rank, msglen, tag); */
    TEAM_UCX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag,
                           dest_group_rank, team->ctx_id);
    ucp_ep_h ep = get_p2p_ep(team, dest_group_rank);
    xccl_ucx_request_t* ucx_req  = (xccl_ucx_request_t *)
        ucp_tag_recv_nb(TEAM_UCX_WORKER(team), buffer, 1, datatype,
                        ucp_tag, ucp_tag_mask, xccl_ucx_recv_completion_cb);
    TEAM_UCX_CHECK_RECV_REQ();
    return XCCL_OK;
}

static inline void xccl_ucx_progress(xccl_ucx_team_t *team)
{
    ucp_worker_progress(TEAM_UCX_WORKER(team));
}

static inline xccl_status_t
xccl_ucx_req_test(xccl_ucx_team_t *team, xccl_ucx_request_t **reqs,
                          int n_reqs, int *completed_idx,
                          int poll_count, int n_completions_required)
{
    int i;
    int n_polls = 0;
    int n_completed;
    assert(NULL != reqs);
    while (poll_count < 0 || n_polls++ < poll_count) {
        n_completed = 0;
        for (i=0; i<n_reqs; i++) {
            if (NULL == reqs[i]) {
                *completed_idx = i;
                n_completed++;
            } else {
                if (reqs[i]->status != XCCL_UCX_REQUEST_DONE) {
                    xccl_ucx_progress(team);
                } else {
                    xccl_ucx_req_free(reqs[i]);
                    reqs[i] = NULL;
                    *completed_idx = i;
                    n_completed++;
                }
            }
            if (n_completed == n_completions_required) {
                return XCCL_OK;
            }
        }
    }
    return XCCL_INPROGRESS;
}

#if 0
static inline
int xccl_ucx_p2p_waitany(xccl_ucx_p2p_request_t **reqs, int n_reqs, int *completed_idx) {
    return xccl_ucx_p2p_test(reqs, n_reqs, completed_idx, -1, 1);
}

static inline
int xccl_ucx_p2p_waitall(xccl_ucx_p2p_request_t **reqs, int n_reqs) {
    int cidx;
    return xccl_ucx_p2p_test(reqs, n_reqs, &cidx, -1, n_reqs);
}
#endif

static inline xccl_status_t
xccl_ucx_team_testany(xccl_ucx_team_t *team, xccl_ucx_request_t **reqs,
                     int n_reqs, int *completed_idx) {
    return xccl_ucx_req_test(team, reqs, n_reqs, completed_idx,
                                 TEAM_UCX_CTX(team)->num_to_probe, 1);
}

static inline xccl_status_t
xccl_ucx_testall(xccl_ucx_team_t *team, xccl_ucx_request_t **reqs,
                     int n_reqs)
{
    int cidx;
    return xccl_ucx_req_test(team, reqs, n_reqs, &cidx,
                                 TEAM_UCX_CTX(team)->num_to_probe, n_reqs);
}
#endif
