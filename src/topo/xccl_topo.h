/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_TOPO_H_
#define XCCL_TOPO_H_

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
#endif
