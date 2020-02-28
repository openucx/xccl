/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef TCCL_TLS_H_
#define TCCL_TLS_H_

typedef enum {
    TCCL_TL_UCX,
    TCCL_TL_HIER,
    TCCL_TL_SHARP,
    TCCL_TL_VMC,
    TCCL_TL_SHMSEG,
    TCCL_TL_LAST
} tccl_tl_id_t;

static inline
const char* tccl_tl_str(tccl_tl_id_t tl_id)
{
    switch(tl_id) {
    case TCCL_TL_UCX:
        return "ucx";
    case TCCL_TL_HIER:
        return "hier";
    case TCCL_TL_SHARP:
        return "sharp";
    case TCCL_TL_VMC:
        return "vmc";
    case TCCL_TL_SHMSEG:
        return "shmseg";
    default:
        break;
    }
    return NULL;
}
#endif
