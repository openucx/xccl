/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef XCCL_HIER_SBGP_H_
#define XCCL_HIER_SBGP_H_
#include "xccl_team_lib.h"
typedef enum sbgp_type_t {
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

typedef struct xccl_hier_team xccl_hier_team_t;
typedef struct sbgp_t {
    sbgp_type_t        type;
    sbgp_status_t      status;
    int                group_size;
    int                group_rank;
    int               *rank_map;
    xccl_hier_team_t  *hier_team;
} sbgp_t;

extern char* sbgp_type_str[SBGP_LAST];
xccl_status_t sbgp_create(xccl_hier_team_t *team, sbgp_type_t type);
xccl_status_t sbgp_cleanup(sbgp_t *sbgp);

static const char* sbgp_status_str(sbgp_status_t status) {
    switch(status) {
    case SBGP_DISABLED:
        return "SBGP_DISABLED";
    case SBGP_ENABLED:
        return "SBGP_ENABLED";
    case SBGP_NOT_EXISTS:
        return "SBGP_NOT_EXIST";
    }
    return NULL;
}

static inline int sbgp_rank2team(sbgp_t *sbgp, int rank)
{
    return sbgp->rank_map[rank];
}

int xccl_hier_compare_proc_data(const void* a, const void* b);
#endif
