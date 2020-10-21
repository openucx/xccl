/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef XCCL_SBGP_H_
#define XCCL_SBGP_H_
#include "xccl_team_lib.h"

typedef enum xccl_sbgp_type_t {
    XCCL_SBGP_UNDEF = 0,
    XCCL_SBGP_NUMA,
    XCCL_SBGP_SOCKET,
    XCCL_SBGP_NODE,
    XCCL_SBGP_NODE_LEADERS,
    XCCL_SBGP_SOCKET_LEADERS,
    XCCL_SBGP_NUMA_LEADERS,
    XCCL_SBGP_FLAT,
    XCCL_SBGP_LAST
} xccl_sbgp_type_t;

typedef enum xccl_sbgp_status_t {
    XCCL_SBGP_NOT_INIT = 0,
    XCCL_SBGP_DISABLED,
    XCCL_SBGP_ENABLED,
    XCCL_SBGP_NOT_EXISTS,
} xccl_sbgp_status_t;

typedef struct xccl_team xccl_team_t;
typedef struct xccl_team_topo xccl_team_topo_t;
typedef struct xccl_sbgp_t {
    xccl_sbgp_type_t   type;
    xccl_sbgp_status_t status;
    int                group_size;
    int                group_rank;
    int               *rank_map;
    xccl_team_t       *team;
} xccl_sbgp_t;

extern char* xccl_sbgp_type_str[XCCL_SBGP_LAST];
xccl_status_t xccl_sbgp_create(xccl_team_topo_t *topo, xccl_sbgp_type_t type);
xccl_status_t xccl_sbgp_cleanup(xccl_sbgp_t *sbgp);

static inline int xccl_sbgp_rank2team(xccl_sbgp_t *sbgp, int rank)
{
    return sbgp->rank_map[rank];
}

void xccl_sbgp_print(xccl_sbgp_t *sbgp);

xccl_status_t xccl_sbgp_oob_allgather(void *sbuf, void *rbuf, size_t len,
                                      xccl_sbgp_t *sbgp, xccl_oob_collectives_t oob);
xccl_status_t xccl_sbgp_oob_bcast(void *buf,size_t len, int root,
                                  xccl_sbgp_t *sbgp, xccl_oob_collectives_t oob);
xccl_status_t xccl_sbgp_oob_barrier(xccl_sbgp_t *sbgp, xccl_oob_collectives_t oob);
#endif
