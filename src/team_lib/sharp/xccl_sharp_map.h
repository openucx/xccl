/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_SHARP_MAP_H_
#define XCCL_SHARP_MAP_H_

#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>

int xccl_to_sharp_dtype[XCCL_DT_LAST_PREDEFINED];
int xccl_to_sharp_reduce_op[XCCL_OP_LAST_PREDEFINED];

static void map_xccl_to_sharp_dtype()
{
    int dt;
    for (dt = 0; dt < XCCL_DT_LAST_PREDEFINED; dt++) {
        xccl_to_sharp_dtype[dt] = SHARP_DTYPE_NULL;
    }
    xccl_to_sharp_dtype[XCCL_DT_INT16]   = SHARP_DTYPE_SHORT;
    xccl_to_sharp_dtype[XCCL_DT_INT32]   = SHARP_DTYPE_INT;
    xccl_to_sharp_dtype[XCCL_DT_INT64]   = SHARP_DTYPE_LONG;
    xccl_to_sharp_dtype[XCCL_DT_UINT16]  = SHARP_DTYPE_UNSIGNED_SHORT;
    xccl_to_sharp_dtype[XCCL_DT_UINT32]  = SHARP_DTYPE_UNSIGNED;
    xccl_to_sharp_dtype[XCCL_DT_UINT64]  = SHARP_DTYPE_UNSIGNED_LONG;
    xccl_to_sharp_dtype[XCCL_DT_FLOAT16] = SHARP_DTYPE_FLOAT_SHORT;
    xccl_to_sharp_dtype[XCCL_DT_FLOAT32] = SHARP_DTYPE_FLOAT;
    xccl_to_sharp_dtype[XCCL_DT_FLOAT64] = SHARP_DTYPE_DOUBLE;
}

static void map_xccl_to_sharp_reduce_op_type()
{
    int op;
    for (op = 0; op < XCCL_OP_LAST_PREDEFINED; op++) {
        xccl_to_sharp_reduce_op[op] = SHARP_OP_NULL;
    }
    xccl_to_sharp_reduce_op[XCCL_OP_MAX]    = SHARP_OP_MAX;
    xccl_to_sharp_reduce_op[XCCL_OP_MIN]    = SHARP_OP_MIN;
    xccl_to_sharp_reduce_op[XCCL_OP_SUM]    = SHARP_OP_SUM;
    xccl_to_sharp_reduce_op[XCCL_OP_PROD]   = SHARP_OP_PROD; 
    /* TODO: not supported?
    xccl_to_sharp_reduce_op[XCCL_OP_AND]    = SHARP_OP_AND;
    xccl_to_sharp_reduce_op[XCCL_OP_OR]     = SHARP_OP_OR;
    xccl_to_sharp_reduce_op[XCCL_OP_XOR]    = SHARP_OP_XOR;*/
    xccl_to_sharp_reduce_op[XCCL_OP_LAND]   = SHARP_OP_LAND;
    xccl_to_sharp_reduce_op[XCCL_OP_LOR]    = SHARP_OP_LOR;
    xccl_to_sharp_reduce_op[XCCL_OP_LXOR]   = SHARP_OP_LXOR;
    xccl_to_sharp_reduce_op[XCCL_OP_BAND]   = SHARP_OP_BAND;
    xccl_to_sharp_reduce_op[XCCL_OP_BOR]    = SHARP_OP_BOR;
    xccl_to_sharp_reduce_op[XCCL_OP_BXOR]   = SHARP_OP_BXOR;
    xccl_to_sharp_reduce_op[XCCL_OP_MAXLOC] = SHARP_OP_MAXLOC;
    xccl_to_sharp_reduce_op[XCCL_OP_MINLOC] = SHARP_OP_MINLOC;
}

#endif