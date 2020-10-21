/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "xccl_topo.h"
#include "xccl_team_lib.h"
#include "utils/utils.h"

#include <limits.h>

int xccl_topo_compare_proc_data(const void* a, const void* b)
{
    const xccl_proc_data_t *d1 = (const xccl_proc_data_t*)a;
    const xccl_proc_data_t *d2 = (const xccl_proc_data_t*)b;
    if (d1->node_hash != d2->node_hash) {
        return d1->node_hash > d2->node_hash ? 1 : -1;
    } else if (d1->socketid != d2->socketid) {
        return d1->socketid - d2->socketid;
    } else {
        return d1->pid - d2->pid;
    }
}

static void compute_layout(xccl_topo_t *topo, xccl_oob_collectives_t oob) {
    int ctx_size = oob.size;
    xccl_proc_data_t *sorted = (xccl_proc_data_t*)
        malloc(ctx_size*sizeof(xccl_proc_data_t));
    memcpy(sorted, topo->procs, ctx_size*sizeof(xccl_proc_data_t));
    qsort(sorted, ctx_size, sizeof(xccl_proc_data_t),
          xccl_topo_compare_proc_data);
    unsigned long current_hash = sorted[0].node_hash;
    int current_ppn = 1;
    int min_ppn = INT_MAX;
    int max_ppn = 0;
    int nnodes = 1;
    int max_sockid = 0;
    int i, j;
    for (i=1; i<ctx_size; i++) {
        unsigned long hash = sorted[i].node_hash;
        if (hash != current_hash) {
            for (j=0; j<ctx_size; j++) {
                if (topo->procs[j].node_hash == current_hash) {
                    topo->procs[j].node_id = nnodes - 1;
                }
            }
            if (current_ppn > max_ppn) max_ppn = current_ppn;
            if (current_ppn < min_ppn) min_ppn = current_ppn;
            nnodes++;
            current_hash = hash;
            current_ppn = 1;
        } else {
            current_ppn++;
        }
    }
    for (j=0; j<ctx_size; j++) {
        if (topo->procs[j].socketid > max_sockid) {
            max_sockid = topo->procs[j].socketid;
        }
        if (topo->procs[j].node_hash == current_hash) {
            topo->procs[j].node_id = nnodes - 1;
        }
    }

    if (current_ppn > max_ppn) max_ppn = current_ppn;
    if (current_ppn < min_ppn) min_ppn = current_ppn;
    free(sorted);
    topo->nnodes = nnodes;
    topo->min_ppn = min_ppn;
    topo->max_ppn = max_ppn;
    topo->max_n_sockets = max_sockid+1;
}

xccl_status_t xccl_topo_init(xccl_oob_collectives_t oob, xccl_topo_t **_topo)
{
    xccl_topo_t *topo = malloc(sizeof(*topo));
    if (!topo) {
        return XCCL_ERR_NO_MEMORY;
    }
    topo->procs = (xccl_proc_data_t*)malloc(
        oob.size*sizeof(xccl_proc_data_t));
    topo->local_proc.socketid  = xccl_local_process_info()->socketid;
    topo->local_proc.node_hash = xccl_local_process_info()->node_hash;
    topo->local_proc.pid       = xccl_local_process_info()->pid;
    topo->n_procs              = oob.size;

    xccl_oob_allgather(&topo->local_proc, topo->procs,
                       sizeof(xccl_proc_data_t), &oob);
    compute_layout(topo, oob);
    *_topo = topo;
    return XCCL_OK;
}

void xccl_topo_cleanup(xccl_topo_t *topo)
{
    free(topo->procs);
    free(topo);
}

xccl_status_t xccl_team_topo_init(xccl_team_t *team, xccl_topo_t *topo, xccl_team_topo_t **_team_topo)
{
    xccl_team_topo_t *team_topo = malloc(sizeof(*team_topo));
    int i;
    if (!topo) {
        return XCCL_ERR_NO_MEMORY;
    }
    team_topo->topo = topo;
    for (i=0; i<XCCL_SBGP_LAST; i++) {
        team_topo->sbgps[i].status = XCCL_SBGP_NOT_INIT;
    }
    team_topo->no_socket = 0;
    team_topo->node_leader_rank = -1;
    team_topo->node_leader_rank_id = 0;
    team_topo->team = team;
    *_team_topo = team_topo;
    return XCCL_OK;
}

void xccl_team_topo_cleanup(xccl_team_topo_t *team_topo)
{
    int i;
    for (i=0; i<XCCL_SBGP_LAST; i++) {
        if (team_topo->sbgps[i].status == XCCL_SBGP_ENABLED) {
            xccl_sbgp_cleanup(&team_topo->sbgps[i]);
        }
    }
    free(team_topo);
}

xccl_sbgp_t* xccl_team_topo_get_sbgp(xccl_team_topo_t *topo, xccl_sbgp_type_t type)
{
    if (topo->sbgps[type].status == XCCL_SBGP_NOT_INIT) {
        xccl_sbgp_create(topo, type);
    }
    return &topo->sbgps[type];
}
