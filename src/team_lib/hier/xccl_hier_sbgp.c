/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include "xccl_hier_sbgp.h"
#include "xccl_hier_team.h"
#include "xccl_hier_context.h"

#define SWAP(_x, _y) do{                        \
        int _tmp   = (_x);                      \
        (_x)       = (_y);                      \
        (_y)       = _tmp;                      \
    } while(0)
enum {
    LOCAL_NODE,
    LOCAL_SOCKET,
};

static inline int is_rank_local(int rank, xccl_hier_team_t *team, int local)
{
    switch (local) {
    case LOCAL_NODE:
        return is_rank_on_local_node(rank, team);
    case LOCAL_SOCKET:
        return is_rank_on_local_socket(rank, team);
    }
}

static inline xccl_status_t sbgp_create_socket(sbgp_t *sbgp)
{
    xccl_hier_team_t *team = sbgp->hier_team;
    sbgp_t *node_sbgp      = &team->sbgps[SBGP_NODE];
    int *local_ranks;
    int group_size     = team->super.params.oob.size;
    int group_rank     = team->super.params.oob.rank;
    int nlr            = team->node_leader_rank;
    int sock_rank = 0, sock_size = 0, i, r, nlr_pos;
    assert(node_sbgp->status == SBGP_ENABLED);
    local_ranks = (int*)malloc(node_sbgp->group_size*sizeof(int));
    if (!local_ranks) {
        return XCCL_ERR_NO_MEMORY;
    }
    for (i=0; i<node_sbgp->group_size; i++) {
        r = node_sbgp->rank_map[i];
        if (is_rank_local(r, team, LOCAL_SOCKET)) {
            local_ranks[sock_size] = r;
            if (r == group_rank) {
                sock_rank = sock_size;
            }
            sock_size++;
        }
    }
    sbgp->group_size = sock_size;
    sbgp->group_rank = sock_rank;
    sbgp->rank_map = local_ranks;
    nlr_pos = -1;
    for (i=0; i<sock_size; i++) {
        if (nlr == local_ranks[i]) {
            nlr_pos = i;
            break;
        }
    }
    if (nlr_pos > 0) {
        if (sock_rank == 0) sbgp->group_rank = nlr_pos;
        if (sock_rank == nlr_pos) sbgp->group_rank = 0;
        SWAP(local_ranks[nlr_pos], local_ranks[0]);
    }
    if (sock_size > 1) {
        sbgp->status = SBGP_ENABLED;
    } else {
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return XCCL_OK;
}

static inline xccl_status_t sbgp_create_node(sbgp_t *sbgp)
{
    xccl_hier_team_t *team   = sbgp->hier_team;
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx,
                                               xccl_hier_context_t);
    int group_size           = team->super.params.oob.size;
    int group_rank           = team->super.params.oob.rank;
    int max_local_size       = 256;
    int ctx_nlr              = ctx->node_leader_rank_id;
    int node_rank = 0, node_size = 0, i;
    int *local_ranks;
    local_ranks = (int*)malloc(max_local_size*sizeof(int));
    if (!local_ranks) {
        return XCCL_ERR_NO_MEMORY;
    }
    for (i=0; i<group_size; i++) {
        if (is_rank_local(i, team, LOCAL_NODE)) {
            if (node_size == max_local_size) {
                max_local_size *= 2;
                local_ranks = (int*)realloc(local_ranks, max_local_size*sizeof(int));
            }
            local_ranks[node_size] = i;

            if (i == group_rank) {
                node_rank = node_size;
            }
            node_size++;
        }
    }
    sbgp->group_size = node_size;
    sbgp->group_rank = node_rank;
    sbgp->rank_map = local_ranks;
    if (0 < ctx_nlr && ctx_nlr < node_size)  {
        /* Rotate local_ranks array so that node_leader_rank_id becomes first
           in that array */
        sbgp->rank_map = (int*)malloc(node_size*sizeof(int));
        if (!sbgp->rank_map) {
            free(local_ranks);
            return XCCL_ERR_NO_MEMORY;
        }
        for (i=ctx_nlr; i<node_size; i++) {
            sbgp->rank_map[i - ctx_nlr] = local_ranks[i];
        }

        for (i=0; i<ctx_nlr; i++) {
            sbgp->rank_map[node_size - ctx_nlr + i] = local_ranks[i];
        }
        sbgp->group_rank = (node_rank + node_size - ctx_nlr) % node_size;
        free(local_ranks);
    }
    team->node_leader_rank = sbgp->rank_map[0];
    if (node_size > 1) {
        sbgp->status = SBGP_ENABLED;
    } else {
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return XCCL_OK;
}

static xccl_status_t sbgp_create_node_leaders(sbgp_t *sbgp)
{
    xccl_hier_team_t *team   = sbgp->hier_team;
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx,
                                               xccl_hier_context_t);
    int comm_size        = team->super.params.oob.size;
    int comm_rank        = team->super.params.oob.rank;
    int ctx_nlr          = ctx->node_leader_rank_id;
    int i_am_node_leader = 0;
    int nnodes           = ctx->nnodes;
    int i, c, n_node_leaders;
    int *nl_array_1, *nl_array_2;

    nl_array_1 = (int*)malloc(nnodes*sizeof(int));
    if (!nl_array_1) {
        return XCCL_ERR_NO_MEMORY;
    }
    nl_array_2 = (int*)malloc(nnodes*sizeof(int));
    if (!nl_array_2) {
        free(nl_array_1);
        return XCCL_ERR_NO_MEMORY;
    }

    for (i=0; i<nnodes; i++) {
        nl_array_1[i] = 0;
        nl_array_2[i] = INT_MAX;
    }

    for (i=0; i<comm_size; i++) {
        int ctx_rank = xccl_hier_team_rank2ctx(team, i);
        int node_id = ctx->procs[ctx_rank].node_id;
        if (nl_array_1[node_id] == 0 ||
            nl_array_1[node_id] == ctx_nlr) {
            nl_array_2[node_id] = i;
        }
        nl_array_1[node_id]++;
    }
    n_node_leaders = 0;
    for (i=0; i<nnodes; i++) {
        if (nl_array_2[i] != INT_MAX) {
            if (comm_rank == nl_array_2[i]) {
                i_am_node_leader = 1;
                sbgp->group_rank = n_node_leaders;
            }
            nl_array_1[n_node_leaders++] = nl_array_2[i];
        }
    }
    free(nl_array_2);

    if (n_node_leaders > 1) {
        if (i_am_node_leader) {
            sbgp->group_size = n_node_leaders;
            sbgp->rank_map = nl_array_1;
            sbgp->status = SBGP_ENABLED;
        } else {
            free(nl_array_1);
            sbgp->status = SBGP_DISABLED;
        }
    } else {
        free(nl_array_1);
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return XCCL_OK;
}

static xccl_status_t sbgp_create_socket_leaders(sbgp_t *sbgp)
{
    xccl_hier_team_t    *team = sbgp->hier_team;
    xccl_hier_context_t *ctx  = ucs_derived_of(team->super.ctx,
                                               xccl_hier_context_t);
    sbgp_t *node_sbgp         = &team->sbgps[SBGP_NODE];
    int    comm_size          = team->super.params.oob.size;
    int    comm_rank          = team->super.params.oob.rank;
    int    nlr                = team->node_leader_rank;
    int    i_am_socket_leader = (nlr == comm_rank);
    int    max_n_sockets      = ctx->max_n_sockets;
    int    *sl_array          = (int*)malloc(max_n_sockets*sizeof(int));
    int    n_socket_leaders   = 1, i, nlr_sock_id;

    if (!sl_array) {
        return XCCL_ERR_NO_MEMORY;
    }
    for (i=0; i<max_n_sockets; i++) {
        sl_array[i] = INT_MAX;
    }
    nlr_sock_id = ctx->procs[xccl_hier_team_rank2ctx(team, nlr)].socketid;
    sl_array[nlr_sock_id] = nlr;

    for (i=0; i<node_sbgp->group_size; i++) {
        int r = node_sbgp->rank_map[i];
        int ctx_rank = xccl_hier_team_rank2ctx(team, r);
        int socket_id = ctx->procs[ctx_rank].socketid;
        if (sl_array[socket_id] == INT_MAX) {
            n_socket_leaders++;
            sl_array[socket_id] = r;
            if (r == comm_rank) {
                i_am_socket_leader = 1;
            }
        }
    }

    if (n_socket_leaders > 1) {
        if (i_am_socket_leader) {
            int sl_rank;
            sbgp->rank_map = (int*)malloc(sizeof(int)*n_socket_leaders);
            if (!sbgp->rank_map) {
                free(sl_array);
                return XCCL_ERR_NO_MEMORY;
            }
            n_socket_leaders = 0;
            for (i=0; i<max_n_sockets; i++) {
                if (sl_array[i] != INT_MAX) {
                    sbgp->rank_map[n_socket_leaders] = sl_array[i];
                    if (comm_rank == sl_array[i]) {
                        sl_rank = n_socket_leaders;
                    }
                    n_socket_leaders++;
                }
            }
            int nlr_pos = -1;
            for (i=0; i<n_socket_leaders; i++) {
                if (sbgp->rank_map[i] == nlr) {
                    nlr_pos = i;
                    break;
                }
            }
            assert(nlr_pos >= 0);
            sbgp->group_rank = sl_rank;
            if (nlr_pos > 0) {
                if (sl_rank == 0) sbgp->group_rank = nlr_pos;
                if (sl_rank == nlr_pos) sbgp->group_rank = 0;
                SWAP(sbgp->rank_map[nlr_pos], sbgp->rank_map[0]);
            }

            sbgp->group_size = n_socket_leaders;
            sbgp->status = SBGP_ENABLED;
        } else {
            sbgp->status = SBGP_DISABLED;
        }
    } else {
        sbgp->status = SBGP_NOT_EXISTS;
    }
    free(sl_array);
    return XCCL_OK;
}

char* sbgp_type_str[SBGP_LAST] = {"undef", "numa", "socket", "node", "node_leaders",
                                  "socket_leaders", "numa_leaders", "flat"};

static void print_sbgp(sbgp_t *sbgp)
{
    int i;
    if (sbgp->group_rank == 0 && sbgp->status == SBGP_ENABLED) {
        printf("sbgp: %15s: group_size %4d, xccl_ranks=[ ",
               sbgp_type_str[sbgp->type], sbgp->group_size);
        for (i=0; i<sbgp->group_size; i++) {
            printf("%d ", sbgp->rank_map[i]);
        }
        printf("]");
        printf("\n");
    }
}

xccl_status_t sbgp_create(xccl_hier_team_t *team, sbgp_type_t type)
{
    xccl_status_t status;
    sbgp_t        *sbgp = &team->sbgps[type];

    sbgp->hier_team = team;
    sbgp->type      = type;
    sbgp->status    = SBGP_NOT_EXISTS;


    switch(type) {
    case SBGP_NODE:
        status = sbgp_create_node(sbgp);
        break;
    case SBGP_SOCKET:
        assert(SBGP_DISABLED != team->sbgps[SBGP_NODE].status);
        if (team->sbgps[SBGP_NODE].status == SBGP_ENABLED) {
            status = sbgp_create_socket(sbgp);
        }
        break;
    case SBGP_NODE_LEADERS:
        assert(SBGP_DISABLED != team->sbgps[SBGP_NODE].status);
        status = sbgp_create_node_leaders(sbgp);
        break;
    case SBGP_SOCKET_LEADERS:
        assert(SBGP_DISABLED != team->sbgps[SBGP_NODE].status);
        if (team->sbgps[SBGP_NODE].status == SBGP_ENABLED) {
            status = sbgp_create_socket_leaders(sbgp);
        }
        break;
    default:
        status = XCCL_ERR_NOT_IMPLEMENTED;
        break;
    };

#if 0
    print_sbgp(sbgp);
#endif
    return status;
}

xccl_status_t sbgp_cleanup(sbgp_t *sbgp)
{
    if (sbgp->rank_map) {
        free(sbgp->rank_map);
    }
    return XCCL_OK;
}

int xccl_hier_compare_proc_data(const void* a, const void* b)
{
    const xccl_hier_proc_data_t *d1 = (const xccl_hier_proc_data_t*)a;
    const xccl_hier_proc_data_t *d2 = (const xccl_hier_proc_data_t*)b;
    if (d1->node_hash != d2->node_hash) {
        return d1->node_hash > d2->node_hash ? 1 : -1;
    } else if (d1->socketid != d2->socketid) {
        return d1->socketid - d2->socketid;
    } else {
        return d1->pid - d2->pid;
    }
}
