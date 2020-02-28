/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "xccl_team_lib.h"

xccl_status_t xccl_team_lib_query(xccl_team_lib_h team_lib,
                                xccl_team_lib_attr_t *attr)
{
    if (attr->field_mask & XCCL_ATTR_FIELD_CONTEXT_CREATE_MODE) {
        attr->context_create_mode = team_lib->ctx_create_mode;
    }
    return XCCL_OK;
}
