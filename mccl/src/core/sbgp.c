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
#include "sbgp.h"
#include "mccl.h"
#include "mccl_inline.h"

enum {
    LOCAL_NODE,
    LOCAL_SOCKET,
};

static inline int is_rank_local(int rank, mccl_comm_t *mccl_comm, int local) {
    switch (local) {
    case LOCAL_NODE:
        return is_rank_on_local_node(rank, mccl_comm);
    case LOCAL_SOCKET:
        return is_rank_on_local_socket(rank, mccl_comm);
    }
}

static inline mccl_status_t sbgp_create_local(sbgp_t *sbgp, int local) {
    mccl_comm_t *mccl_comm = sbgp->mccl_comm;
    int *local_ranks;
    int group_size     = mccl_comm->config.comm_size;
    int group_rank     = mccl_comm->config.comm_rank;
    int max_local_size = 256;
    int node_rank = 0, node_size = 0, i;
    local_ranks = (int*)malloc(max_local_size*sizeof(int));

    for (i=0; i<group_size; i++) {
        if (is_rank_local(i, mccl_comm, local)) {
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
    sbgp->mccl_rank_map = local_ranks;
    if (node_size > 1) {
        sbgp->status = SBGP_ENABLED;
    } else {
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return MCCL_SUCCESS;
}

static mccl_status_t sbgp_create_node(sbgp_t *sbgp) {
    return sbgp_create_local(sbgp, LOCAL_NODE);
}

static mccl_status_t sbgp_create_socket(sbgp_t *sbgp) {
    return sbgp_create_local(sbgp, LOCAL_SOCKET);
}

static mccl_status_t sbgp_create_node_leaders(sbgp_t *sbgp) {
    mccl_comm_t *mccl_comm = sbgp->mccl_comm;
    mccl_context_t* mccl_ctx = (mccl_context_t*)mccl_comm->config.mccl_ctx;
    int comm_size     = mccl_comm->config.comm_size;
    int comm_rank     = mccl_comm->config.comm_rank;
    int i, c;
    int i_am_node_leader = 0;
    int nnodes = mccl_ctx->nnodes;
    int *nl_array_1 = (int*)malloc(nnodes*sizeof(int));
    int *nl_array_2 = (int*)malloc(nnodes*sizeof(int));
    int n_node_leaders;

    for (i=0; i<nnodes; i++) {
        nl_array_1[i] = INT_MAX;
        nl_array_2[i] = INT_MAX;
    }

    for (i=0; i<comm_size; i++) {
        int world_rank = mccl_comm_rank2world(mccl_comm, i);
        int node_id = mccl_ctx->procs[world_rank].node_id;
        if (nl_array_1[node_id] > world_rank) {
            nl_array_1[node_id] = world_rank;
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
            sbgp->mccl_rank_map = nl_array_1;
            sbgp->status = SBGP_ENABLED;
        } else {
            free(nl_array_1);
            sbgp->status = SBGP_DISABLED;
        }
    } else {
        free(nl_array_1);
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return MCCL_SUCCESS;
}

static mccl_status_t sbgp_create_socket_leaders(sbgp_t *sbgp) {
    mccl_comm_t *mccl_comm = sbgp->mccl_comm;
    mccl_context_t* mccl_ctx = (mccl_context_t*)mccl_comm->config.mccl_ctx;
    int comm_size     = mccl_comm->config.comm_size;
    int comm_rank     = mccl_comm->config.comm_rank;
    int i, c;
    int i_am_socket_leader = 0;
    int max_ppn = mccl_ctx->max_ppn;//TODO can be changed to max_sockets_per_node
    int *sl_array_1 = (int*)malloc(max_ppn*sizeof(int));
    int *sl_array_2 = (int*)malloc(max_ppn*sizeof(int));
    int n_socket_leaders;
    unsigned long my_node_hash = mccl_ctx->local_proc.node_hash;
    for (i=0; i<max_ppn; i++) {
        sl_array_1[i] = INT_MAX;
        sl_array_2[i] = INT_MAX;
    }

     for (i=0; i<comm_size; i++) {
        int world_rank = mccl_comm_rank2world(mccl_comm, i);
        int socket_id = mccl_ctx->procs[world_rank].socketid;
        unsigned long node_hash = mccl_ctx->procs[world_rank].node_hash;
        if (node_hash == my_node_hash &&
            sl_array_1[socket_id] > world_rank) {
            sl_array_1[socket_id] = world_rank;
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
            sbgp->mccl_rank_map = sl_array_1;
            sbgp->status = SBGP_ENABLED;
        } else {
            free(sl_array_1);
            sbgp->status = SBGP_DISABLED;
        }
    } else {
        free(sl_array_1);
        sbgp->status = SBGP_NOT_EXISTS;
    }
    return MCCL_SUCCESS;
}

char* sbgp_type_str[SBGP_LAST] = {"undef", "numa", "socket", "node", "node_leaders",
                                  "socket_leaders", "numa_leaders", "flat"};

static void print_sbgp(sbgp_t *sbgp) {
    int i;
    if (sbgp->group_rank == 0 && sbgp->status == SBGP_ENABLED) {
        printf("sbgp \"%s\": group_size %d, mccl_ranks=[", sbgp_type_str[sbgp->type],
               sbgp->group_size);
        for (i=0; i<sbgp->group_size; i++) {
            printf("%d ", sbgp->mccl_rank_map[i]);
        }
        printf("]");
        printf("\n");
    }
}

mccl_status_t sbgp_create(mccl_comm_t *mccl_comm, sbgp_type_t type, sbgp_t *sbgp) {
    mccl_status_t ret;
    sbgp->mccl_comm = mccl_comm;
    sbgp->type = type;
    switch(type) {
    case SBGP_NODE:
        ret = sbgp_create_node(sbgp);
        break;
    case SBGP_SOCKET:
        ret = sbgp_create_socket(sbgp);
        break;
    case SBGP_NODE_LEADERS:
        ret = sbgp_create_node_leaders(sbgp);
        break;
    case SBGP_SOCKET_LEADERS:
        ret = sbgp_create_socket_leaders(sbgp);
        break;
    default:
        ret = MCCL_ERROR;
        printf("not implemented\n");
        break;
    };

#if 0
    print_sbgp(sbgp);
#endif
    return ret;
}

mccl_status_t sbgp_cleanup(sbgp_t *sbgp) {
    if (sbgp->mccl_rank_map) {
        free(sbgp->mccl_rank_map);
    }
    return MCCL_SUCCESS;
}
