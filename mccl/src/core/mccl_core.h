/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_CORE_H
#define MCCL_CORE_H
#include <stdint.h>
#include "sbgp.h"
#include "mccl.h"
#include "api/tccl.h"

typedef struct mccl_team_t mccl_team_t;

typedef struct proc_data {
    unsigned long node_hash;
    int node_id;
    int socketid; //if process is bound to a socket
} proc_data_t;

typedef enum {
    TCCL_LIB_UCX,
    TCCL_LIB_SHMSEG,
    TCCL_LIB_SHARP,
    TCCL_LIB_VMC,
    TCCL_LIB_LAST
} mccl_tccl_team_lib_t;

typedef struct mccl_team_lib_t {
    tccl_context_h tccl_ctx;
    int            enabled;
} mccl_team_lib_t;

typedef struct mccl_context_t {
    mccl_config_t   config;
    tccl_lib_h      tccl_lib;
    proc_data_t     local_proc; // local proc data
    proc_data_t    *procs; // data for all processes
    int             nnodes;
    int             min_ppn;
    int             max_ppn;
    int             max_sockets_per_node;
    int             max_ranks_per_socket;
    mccl_team_lib_t libs[TCCL_LIB_LAST];
} mccl_context_t;

typedef enum {
    MCCL_TEAM_NODE_UCX,
    MCCL_TEAM_SOCKET_UCX,
    MCCL_TEAM_NODE_LEADERS_UCX,
    MCCL_TEAM_SOCKET_LEADERS_UCX,
    MCCL_TEAM_NODE_SHMSEG,
    MCCL_TEAM_SOCKET_SHMSEG,
    MCCL_TEAM_SOCKET_LEADERS_SHMSEG,
    MCCL_TEAM_NODE_LEADERS_SHARP,
    MCCL_TEAM_NODE_LEADERS_VMC,
    MCCL_TEAM_LAST,
} mccl_team_type_t;

typedef struct mccl_comm_t {
    mccl_comm_config_t config;
    int               *world_ranks; // map of local comm ranks to world ranks
    sbgp_t             sbgps[SBGP_LAST];
    mccl_team_t       *teams[MCCL_TEAM_LAST];
    void              *static_team_data[MCCL_TEAM_LAST];
    int64_t            seq_num;
    int                ctx_id; //TODO
} mccl_comm_t;

mccl_status_t mccl_get_bound_socket_id(int *socketid);
#endif
