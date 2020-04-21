/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_MM_H_
#define XCCL_MM_H_

#include <api/xccl.h>
#include <xccl_team.h>

typedef struct xccl_tl_mem_handle {
    xccl_tl_id_t id;
} xccl_tl_mem_handle_t;

typedef struct xccl_mem_handle {
    xccl_team_t   *team;
    xccl_tl_mem_h handles[1];
} xccl_mem_handle_t;

static inline xccl_tl_mem_h xccl_mem_handle_by_tl_id(xccl_mem_h memh, xccl_tl_id_t id)
{
    int i;

    for (i=0; i<memh->team->n_teams; i++) {
        if (memh->handles[i]->id == id) {
            return memh->handles[i];
        }
    }

    return NULL;
}

#endif
