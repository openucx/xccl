/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TOPO_H_
#define XCCL_TOPO_H_
#include "xccl_sbgp.h"

typedef struct xccl_proc_data {
    unsigned long node_hash;
    int           node_id;
    int           socketid; //if process is bound to a socket
    int           pid;
} xccl_proc_data_t;

typedef struct xccl_topo {
    xccl_proc_data_t local_proc;
    xccl_proc_data_t *procs;
    int              n_procs;
    int              nnodes;
    int              min_ppn;
    int              max_ppn;
    int              max_n_sockets;
} xccl_topo_t;

typedef struct xccl_team xccl_team_t;
typedef struct xccl_team_topo {
    xccl_topo_t *topo;
    xccl_sbgp_t  sbgps[XCCL_SBGP_LAST];
    int          node_leader_rank;
    int          node_leader_rank_id;
    int          no_socket;
    xccl_team_t *team;
} xccl_team_topo_t;

xccl_status_t xccl_topo_init(xccl_oob_collectives_t oob, xccl_topo_t **topo);
void xccl_topo_cleanup(xccl_topo_t *topo);

xccl_status_t xccl_team_topo_init(xccl_team_t *team, xccl_topo_t *topo,
                                  xccl_team_topo_t **team_topo);
void xccl_team_topo_cleanup(xccl_team_topo_t *team_topo);
xccl_sbgp_t* xccl_team_topo_get_sbgp(xccl_team_topo_t *topo, xccl_sbgp_type_t type);

#endif
