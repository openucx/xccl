/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TCCL_SHARP_MAP_H_
#define TCCL_SHARP_MAP_H_

#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>

int tccl_to_sharp_dtype[TCCL_DT_LAST_PREDEFINED];
int tccl_to_sharp_reduce_op[TCCL_OP_LAST_PREDEFINED];

static void map_tccl_to_sharp_dtype()
{
    int dt;
    for (dt = 0; dt < TCCL_DT_LAST_PREDEFINED; dt++) {
        tccl_to_sharp_dtype[dt] = SHARP_DTYPE_NULL;
    }
    tccl_to_sharp_dtype[TCCL_DT_INT16]   = SHARP_DTYPE_SHORT;
    tccl_to_sharp_dtype[TCCL_DT_INT32]   = SHARP_DTYPE_INT;
    tccl_to_sharp_dtype[TCCL_DT_INT64]   = SHARP_DTYPE_LONG;
    tccl_to_sharp_dtype[TCCL_DT_UINT16]  = SHARP_DTYPE_UNSIGNED_SHORT;
    tccl_to_sharp_dtype[TCCL_DT_UINT32]  = SHARP_DTYPE_UNSIGNED;
    tccl_to_sharp_dtype[TCCL_DT_UINT64]  = SHARP_DTYPE_UNSIGNED_LONG;
    tccl_to_sharp_dtype[TCCL_DT_FLOAT16] = SHARP_DTYPE_FLOAT_SHORT;
    tccl_to_sharp_dtype[TCCL_DT_FLOAT32] = SHARP_DTYPE_FLOAT;
    tccl_to_sharp_dtype[TCCL_DT_FLOAT64] = SHARP_DTYPE_DOUBLE;
}

static void map_tccl_to_sharp_reduce_op_type()
{
    int op;
    for (op = 0; op < TCCL_OP_LAST_PREDEFINED; op++) {
        tccl_to_sharp_reduce_op[op] = SHARP_OP_NULL;
    }
    tccl_to_sharp_reduce_op[TCCL_OP_MAX]    = SHARP_OP_MAX;
    tccl_to_sharp_reduce_op[TCCL_OP_MIN]    = SHARP_OP_MIN;
    tccl_to_sharp_reduce_op[TCCL_OP_SUM]    = SHARP_OP_SUM;
    tccl_to_sharp_reduce_op[TCCL_OP_PROD]   = SHARP_OP_PROD; 
    /* TODO: not supported?
    tccl_to_sharp_reduce_op[TCCL_OP_AND]    = SHARP_OP_AND;
    tccl_to_sharp_reduce_op[TCCL_OP_OR]     = SHARP_OP_OR;
    tccl_to_sharp_reduce_op[TCCL_OP_XOR]    = SHARP_OP_XOR;*/
    tccl_to_sharp_reduce_op[TCCL_OP_LAND]   = SHARP_OP_LAND;
    tccl_to_sharp_reduce_op[TCCL_OP_LOR]    = SHARP_OP_LOR;
    tccl_to_sharp_reduce_op[TCCL_OP_LXOR]   = SHARP_OP_LXOR;
    tccl_to_sharp_reduce_op[TCCL_OP_BAND]   = SHARP_OP_BAND;
    tccl_to_sharp_reduce_op[TCCL_OP_BOR]    = SHARP_OP_BOR;
    tccl_to_sharp_reduce_op[TCCL_OP_BXOR]   = SHARP_OP_BXOR;
    tccl_to_sharp_reduce_op[TCCL_OP_MAXLOC] = SHARP_OP_MAXLOC;
    tccl_to_sharp_reduce_op[TCCL_OP_MINLOC] = SHARP_OP_MINLOC;
}

#endif