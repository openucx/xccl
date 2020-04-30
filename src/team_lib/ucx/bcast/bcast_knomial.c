#include "config.h"
#include "xccl_ucx_lib.h"
#include "bcast.h"
#include "xccl_ucx_sendrecv.h"
#include <stdlib.h>
#include <string.h>

#define CALC_DIST(_size, _radix, _dist) do{     \
        _dist = 1;                              \
        while (_dist*_radix < _size) {          \
            _dist*=_radix;                      \
        }                                       \
    }while(0)

xccl_status_t xccl_ucx_bcast_knomial_progress(xccl_ucx_collreq_t *req)
{
    xccl_tl_team_t *team = req->team;
    void *data_buffer    = req->args.buffer_info.dst_buffer;
    size_t data_size     = req->args.buffer_info.len;
    int group_rank       = team->params.oob.rank;
    int group_size       = team->params.oob.size;
    int root             = req->args.root;
    int radix            = req->bcast_kn.radix;
    xccl_ucx_request_t **reqs = req->bcast_kn.reqs;
    int vrank = (group_rank - root + group_size) % group_size;
    int dist  = req->bcast_kn.dist;
    int i, vpeer, peer, vroot_at_level, root_at_level, pos;

    if (req->bcast_kn.active_reqs) {
        if (XCCL_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs,
                                           req->bcast_kn.active_reqs)) {
            req->bcast_kn.active_reqs = 0;
        } else {
            return XCCL_OK;
        }
    }

    while (dist >= 1) {
        if (vrank % dist == 0) {
            pos = (vrank/dist) % radix;
        } else {
            pos = -1;
        }
        if (pos == 0) {
            for (i=radix-1; i>=1; i--) {
                vpeer = vrank + i*dist;
                if (vpeer < group_size) {
                    peer = (vpeer + root) % group_size;
                    xccl_ucx_send_nb(data_buffer, data_size, peer,
                                    (xccl_ucx_team_t*)team, req->tag,
                                    &reqs[req->bcast_kn.active_reqs++]);
                }
            }
        } else if (pos > 0) {
            vroot_at_level = vrank - pos*dist;
            root_at_level  = (vroot_at_level + root) % group_size;
            xccl_ucx_recv_nb(data_buffer, data_size, root_at_level,
                            (xccl_ucx_team_t*)team, req->tag, &reqs[req->bcast_kn.active_reqs++]);
            assert(req->bcast_kn.active_reqs == 1);
        }
        dist /= radix;

        if (req->bcast_kn.active_reqs) {
            if (XCCL_OK == xccl_ucx_testall((xccl_ucx_team_t *)team, reqs,
                                               req->bcast_kn.active_reqs)) {
                req->bcast_kn.active_reqs = 0;
            } else {
                req->bcast_kn.dist = dist;
                return XCCL_OK;
            }
        }
    }
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_bcast_knomial_start(xccl_ucx_collreq_t *req)
{
    size_t data_size = req->args.buffer_info.len;
    int group_rank   = req->team->params.oob.rank;
    int group_size   = req->team->params.oob.size;
    xccl_ucx_debug("knomial bcast start: group_size %d, group_rank %d, data_size %zd",
                   group_size, group_rank, data_size);
    memset(req->bcast_kn.reqs, 0, sizeof(req->bcast_kn.reqs));
    req->bcast_kn.radix   = 4;//TODO
    if (req->bcast_kn.radix > req->team->params.oob.size) {
        req->bcast_kn.radix = req->team->params.oob.size;
    }

    req->bcast_kn.active_reqs = 0;
    CALC_DIST(group_size, req->bcast_kn.radix, req->bcast_kn.dist);
    if (req->args.root == group_rank) {
        if (req->args.buffer_info.src_buffer !=
            req->args.buffer_info.dst_buffer) {
            xccl_ucx_send_recv(req->args.buffer_info.src_buffer, data_size,
                               group_rank, req->tag, req->args.buffer_info.dst_buffer,
                               data_size, group_rank, req->tag,
                               (xccl_ucx_team_t *)req->team);
        }
    }
    req->progress = xccl_ucx_bcast_knomial_progress;
    return xccl_ucx_bcast_knomial_progress(req);
}
