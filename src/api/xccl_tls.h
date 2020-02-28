/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_TLS_H_
#define XCCL_TLS_H_

typedef enum {
    XCCL_TL_UCX,
    XCCL_TL_HIER,
    XCCL_TL_SHARP,
    XCCL_TL_VMC,
    XCCL_TL_SHMSEG,
    XCCL_TL_LAST
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
#endif
