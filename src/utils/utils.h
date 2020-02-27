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
                   tccl_tl_team_t *team, tccl_oob_collectives_t oob)
{
    size_t team_size = oob.size;
    void *tmp;
    size_t len = count*tccl_dt_size(dt);
    int i;
    tmp = malloc(team_size*len);
    tccl_oob_allgather(sbuf, tmp, len, &oob);
    tccl_dt_reduce(tmp, (void*)((ptrdiff_t)tmp + len), rbuf, count, dt, op);
    for (i=2; i<team_size; i++) {
        tccl_dt_reduce(rbuf, (void*)((ptrdiff_t)tmp + i*len), rbuf, count, dt, op);
    }
    free(tmp);
}

tccl_status_t tccl_get_bound_socket_id(int *socketid);

static inline
unsigned long tccl_str_hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while (c = *str++) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

#endif
