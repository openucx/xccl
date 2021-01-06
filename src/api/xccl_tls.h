/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_TLS_H_
#define XCCL_TLS_H_

#include <ucs/config/types.h>
#include <string.h>

typedef enum xccl_tl_id {
    XCCL_TL_NULL   = 0,
    XCCL_TL_UCX    = UCS_BIT(0),
    XCCL_TL_HIER   = UCS_BIT(1),
    XCCL_TL_SHARP  = UCS_BIT(2),
    XCCL_TL_HMC    = UCS_BIT(3),
    XCCL_TL_SHMSEG = UCS_BIT(4),
    XCCL_TL_MRAIL  = UCS_BIT(5),
    XCCL_TL_NCCL   = UCS_BIT(6),
    XCCL_TL_DPU    = UCS_BIT(7),
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
    case XCCL_TL_MRAIL:
        return "mrail";
    case XCCL_TL_SHARP:
        return "sharp";
    case XCCL_TL_HMC:
        return "hmc";
    case XCCL_TL_SHMSEG:
        return "shmseg";
    case XCCL_TL_NCCL:
        return "nccl";
    case XCCL_TL_DPU:
        return "dpu";
    default:
        break;
    }
    return NULL;
}

static inline
xccl_tl_id_t xccl_tls_str_to_bitmap(const char *tls_str)
{
    xccl_tl_id_t tls = XCCL_TL_NULL;
    uint64_t     i;

    if (!tls_str) {
        return tls;
    }

    for (i = 1; i < XCCL_TL_LAST; i = i << 1) {
        if (strstr(tls_str, xccl_tl_str((xccl_tl_id_t)i))) {
            tls = (xccl_tl_id_t)(tls | i);
        }
    }

    return tls;
}

#endif
