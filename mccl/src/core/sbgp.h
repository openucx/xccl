/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef SBGP_H
#define SBGP_H
#include <api/mccl.h>
typedef enum sbgp_type_t {
    SBGP_UNDEF = 0,
    SBGP_NUMA,
    SBGP_SOCKET,
    SBGP_NODE,
    SBGP_NODE_LEADERS,
    SBGP_SOCKET_LEADERS,
    SBGP_NUMA_LEADERS,
    SBGP_FLAT,
    SBGP_LAST
} sbgp_type_t;

typedef enum sbgp_status_t {
    SBGP_DISABLED = 0,
    SBGP_ENABLED,
    SBGP_NOT_EXISTS,
} sbgp_status_t;

typedef struct mccl_comm_t mccl_comm_t;
typedef struct sbgp_t {
    sbgp_type_t type;
    sbgp_status_t status;
    int group_size;
    int group_rank;
    int *mccl_rank_map;
    mccl_comm_t *mccl_comm;
} sbgp_t;

extern char* sbgp_type_str[SBGP_LAST];

mccl_status_t sbgp_create(mccl_comm_t *mccl_comm, sbgp_type_t type, sbgp_t *sbgp);
mccl_status_t sbgp_cleanup(sbgp_t *sbgp);
#endif
