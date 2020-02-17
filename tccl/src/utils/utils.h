/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef TCCL_UTILS_H_
#define TCCL_UTILS_H
#include "api/tccl.h"
#include "reduce.h"
#include <stdlib.h>

static inline void
tccl_oob_allreduce(void *sbuf, void *rbuf, size_t count, tccl_dt_t dt, tccl_op_t op,
                  tccl_team_h team, tccl_oob_collectives_t oob)
{
    size_t team_size = team->cfg.team_size;
    void *tmp;
    size_t len = count*tccl_dt_size(dt);
    int i;
    tmp = malloc(team_size*len);
    oob.allgather(sbuf, tmp, len, oob.coll_context);
    tccl_dt_reduce(tmp, (void*)((ptrdiff_t)tmp + len), rbuf, count, dt, op);
    for (i=2; i<team_size; i++) {
        tccl_dt_reduce(rbuf, (void*)((ptrdiff_t)tmp + i*len), rbuf, count, dt, op);
    }
    free(tmp);
}
#endif
