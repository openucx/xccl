/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_TLS_H_
#define XCCL_TLS_H_

#include <ucs/config/types.h>

typedef enum xccl_tl_id {
    XCCL_TL_UCX    = UCS_BIT(0),
    XCCL_TL_HIER   = UCS_BIT(1),
    XCCL_TL_SHARP  = UCS_BIT(2),
    XCCL_TL_VMC    = UCS_BIT(3),
    XCCL_TL_SHMSEG = UCS_BIT(4),
    XCCL_TL_LAST,
    XCCL_TL_ALL    = (XCCL_TL_LAST << 1) - 3
} xccl_tl_id_t;

static inline
const char* xccl_tl_str(xccl_tl_id_t tl_id)
{
    switch(tl_id) {
    case XCCL_TL_UCX:
        return "ucx";
    case XCCL_TL_HIER:
        return "hier";
    case XCCL_TL_SHARP:
        return "sharp";
    case XCCL_TL_VMC:
        return "vmc";
    case XCCL_TL_SHMSEG:
        return "shmseg";
    default:
        break;
    }
    return NULL;
}

static inline
xccl_tl_id_t xccl_tls_str_to_bitmap(const char *tls_str)
{
    xccl_tl_id_t tls = 0;
    int i;

    if (!tls_str) {
        return tls;
    }

    for (i = 1; i < XCCL_TL_LAST; i = i << 1) {
        if (strstr(tls_str, xccl_tl_str(i))) {
            tls = tls | i;
        } 
    }

    return tls;
}

#endif
