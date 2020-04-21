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

enum {
    LOCAL_NODE,
    LOCAL_SOCKET,
};

static inline int is_rank_local(int rank, xccl_hier_team_t *team, int local) {
    switch (local) {
    case LOCAL_NODE:
        return is_rank_on_local_node(rank, team);
    case LOCAL_SOCKET:
        return is_rank_on_local_socket(rank, team);
    }
}

static inline xccl_status_t sbgp_create_local(sbgp_t *sbgp, int local) {
    xccl_hier_team_t *team = sbgp->hier_team;
    int *local_ranks;
    int group_size     = team->super.params.oob.size;
    int group_rank     = team->super.params.oob.rank;
    int max_local_size = 256;
    int node_rank = 0, node_size = 0, i;
    local_ranks = (int*)malloc(max_local_size*sizeof(int));

    for (i=0; i<group_size; i++) {
        if (is_rank_local(i, team, local)) {
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
    if (node_size > 1) {
        sbgp->status = SBGP_ENABLED;
    } else {
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return XCCL_OK;
}

static xccl_status_t sbgp_create_node(sbgp_t *sbgp) {
    return sbgp_create_local(sbgp, LOCAL_NODE);
}

static xccl_status_t sbgp_create_socket(sbgp_t *sbgp) {
    return sbgp_create_local(sbgp, LOCAL_SOCKET);
}

static xccl_status_t sbgp_create_node_leaders(sbgp_t *sbgp) {
    xccl_hier_team_t *team = sbgp->hier_team;
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx,
                                               xccl_hier_context_t);
    int comm_size     = team->super.params.oob.size;
    int comm_rank     = team->super.params.oob.rank;
    int i, c;
    int i_am_node_leader = 0;
    int nnodes = ctx->nnodes;
    int *nl_array_1 = (int*)malloc(nnodes*sizeof(int));
    int *nl_array_2 = (int*)malloc(nnodes*sizeof(int));
    int n_node_leaders;

    for (i=0; i<nnodes; i++) {
        nl_array_1[i] = INT_MAX;
        nl_array_2[i] = INT_MAX;
    }

    for (i=0; i<comm_size; i++) {
        int ctx_rank = xccl_hier_team_rank2ctx(team, i);
        int node_id = ctx->procs[ctx_rank].node_id;
        if (nl_array_1[node_id] > ctx_rank) {
            nl_array_1[node_id] = ctx_rank;
            nl_array_2[node_id] = i;
        }
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

static xccl_status_t sbgp_create_socket_leaders(sbgp_t *sbgp) {
    xccl_hier_team_t *team = sbgp->hier_team;
    xccl_hier_context_t *ctx = ucs_derived_of(team->super.ctx,
                                               xccl_hier_context_t);
    int comm_size     = team->super.params.oob.size;
    int comm_rank     = team->super.params.oob.rank;
    int i, c;
    int i_am_socket_leader = 0;
    int max_ppn = ctx->max_ppn;//TODO can be changed to max_sockets_per_node
    int *sl_array_1 = (int*)malloc(max_ppn*sizeof(int));
    int *sl_array_2 = (int*)malloc(max_ppn*sizeof(int));
    int n_socket_leaders;
    unsigned long my_node_hash = ctx->local_proc.node_hash;
    for (i=0; i<max_ppn; i++) {
        sl_array_1[i] = INT_MAX;
        sl_array_2[i] = INT_MAX;
    }

     for (i=0; i<comm_size; i++) {
        int ctx_rank = xccl_hier_team_rank2ctx(team, i);
        int socket_id = ctx->procs[ctx_rank].socketid;
        unsigned long node_hash = ctx->procs[ctx_rank].node_hash;
        if (node_hash == my_node_hash &&
            sl_array_1[socket_id] > ctx_rank) {
            sl_array_1[socket_id] = ctx_rank;
            sl_array_2[socket_id] = i;
        }
    }
    n_socket_leaders = 0;
    for (i=0; i<max_ppn; i++) {
        if (sl_array_2[i] != INT_MAX) {
            if (comm_rank == sl_array_2[i]) {
                i_am_socket_leader = 1;
                sbgp->group_rank = n_socket_leaders;
            }
            sl_array_1[n_socket_leaders++] = sl_array_2[i];
        }
    }
    free(sl_array_2);

    if (n_socket_leaders > 1) {
        if (i_am_socket_leader) {
            sbgp->group_size = n_socket_leaders;
            sbgp->rank_map = sl_array_1;
            sbgp->status = SBGP_ENABLED;
        } else {
            free(sl_array_1);
            sbgp->status = SBGP_DISABLED;
        }
    } else {
        free(sl_array_1);
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return XCCL_OK;
}

char* sbgp_type_str[SBGP_LAST] = {"undef", "numa", "socket", "node", "node_leaders",
                                  "socket_leaders", "numa_leaders", "flat"};

static void print_sbgp(sbgp_t *sbgp) {
    int i;
    if (sbgp->group_rank == 0 && sbgp->status == SBGP_ENABLED) {
        printf("sbgp \"%s\": group_size %d, xccl_ranks=[", sbgp_type_str[sbgp->type],
               sbgp->group_size);
        for (i=0; i<sbgp->group_size; i++) {
            printf("%d ", sbgp->rank_map[i]);
        }
        printf("]");
        printf("\n");
    }
}

xccl_status_t sbgp_create(xccl_hier_team_t *team, sbgp_type_t type) {
    xccl_status_t status;
    sbgp_t *sbgp = &team->sbgps[type];
    sbgp->hier_team = team;
    sbgp->type = type;
    switch(type) {
    case SBGP_NODE:
        status = sbgp_create_node(sbgp);
        break;
    case SBGP_SOCKET:
        status = sbgp_create_socket(sbgp);
        break;
    case SBGP_NODE_LEADERS:
        status = sbgp_create_node_leaders(sbgp);
        break;
    case SBGP_SOCKET_LEADERS:
        status = sbgp_create_socket_leaders(sbgp);
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

xccl_status_t sbgp_cleanup(sbgp_t *sbgp) {
    if (sbgp->rank_map) {
        free(sbgp->rank_map);
    }
    return XCCL_OK;
}
