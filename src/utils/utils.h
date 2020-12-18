/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef XCCL_UTILS_H_
#define XCCL_UTILS_H
#include "api/xccl.h"
#include "reduce.h"
#include <stdlib.h>

static inline void
xccl_oob_allreduce(void *sbuf, void *rbuf, size_t count, xccl_dt_t dt, xccl_op_t op,
                   xccl_tl_team_t *team, xccl_oob_collectives_t oob)
{
    size_t team_size = oob.size;
    void *tmp;
    size_t len = count*xccl_dt_size(dt);
    int i;
    tmp = malloc(team_size*len);
    xccl_oob_allgather(sbuf, tmp, len, &oob);
    xccl_dt_reduce(tmp, (void*)((ptrdiff_t)tmp + len), rbuf, count, dt, op);
    for (i=2; i<team_size; i++) {
        xccl_dt_reduce(rbuf, (void*)((ptrdiff_t)tmp + i*len), rbuf, count, dt, op);
    }
    free(tmp);
}

xccl_status_t xccl_get_bound_socket_id(int *socketid);

static inline
unsigned long xccl_str_hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while (c = *str++) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

static inline char*
xccl_names_array_to_str(ucs_config_names_array_t *array) {
    size_t total_len = 0;
    int    i;
    char   *str;
    if (array->count == 0) {
        return NULL;
    }
    for (i=0; i<array->count; i++) {
        total_len += strlen(array->names[i]) + 1;
    }
    str = (char*)malloc(total_len);
    if (!str) {
        return NULL;
    }
    strcpy(str, array->names[0]);
    for (i=1; i<array->count; i++) {
        strcat(str, ",");
        strcat(str, array->names[i]);
    }
    return str;
}

static inline int xccl_round_up(int dividend, int divisor) {
    return (dividend + divisor - 1) / divisor;
}

#endif
