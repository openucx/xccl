/*
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "xccl_sbgp.h"
#include "xccl_topo.h"
#include "core/xccl_team.h"
char* xccl_sbgp_type_str[XCCL_SBGP_LAST] = {"undef", "numa", "socket", "node", "node_leaders",
                                            "socket_leaders", "numa_leaders", "flat"};

enum {
    LOCAL_NODE,
    LOCAL_SOCKET,
};


static inline xccl_status_t sbgp_create_node(xccl_team_topo_t *topo, xccl_sbgp_t *sbgp)
{
    xccl_team_t *team        = sbgp->team;
    int group_size           = team->params.oob.size;
    int group_rank           = team->params.oob.rank;
    int max_local_size       = 256;
    int ctx_nlr              = topo->node_leader_rank_id;
    int node_rank = 0, node_size = 0, i;
    int *local_ranks;
    local_ranks = (int*)malloc(max_local_size*sizeof(int));
    if (!local_ranks) {
        return XCCL_ERR_NO_MEMORY;
    }
    for (i=0; i<group_size; i++) {
        if (xccl_rank_on_local_node(i, team)) {
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
    topo->node_leader_rank = sbgp->rank_map[0];
    if (node_size > 1) {
        sbgp->status = XCCL_SBGP_ENABLED;
    } else {
        sbgp->status = XCCL_SBGP_NOT_EXISTS;
    }
    return XCCL_OK;
}

xccl_status_t xccl_sbgp_create(xccl_team_topo_t *topo, xccl_sbgp_type_t type)
{
    xccl_status_t status;
    xccl_team_t *team = topo->team;
    xccl_sbgp_t  *sbgp = &topo->sbgps[type];

    sbgp->team   = team;
    sbgp->type   = type;
    sbgp->status = XCCL_SBGP_NOT_EXISTS;

    switch(type) {
    case XCCL_SBGP_NODE:
        status = sbgp_create_node(topo, sbgp);
        break;
#if 0
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
#endif
    default:
        status = XCCL_ERR_NOT_IMPLEMENTED;
        break;
    };

#if 0
    print_sbgp(sbgp);
#endif

}

xccl_status_t xccl_sbgp_cleanup(xccl_sbgp_t *sbgp)
{
    if (sbgp->rank_map) {
        free(sbgp->rank_map);
    }
}
