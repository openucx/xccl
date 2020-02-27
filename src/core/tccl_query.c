/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "config.h"
#include "tccl_team_lib.h"

tccl_status_t tccl_team_lib_query(tccl_team_lib_h team_lib,
                                tccl_team_lib_attr_t *attr)
{
    if (attr->field_mask & TCCL_ATTR_FIELD_CONTEXT_CREATE_MODE) {
        attr->context_create_mode = team_lib->ctx_create_mode;
    }
    return TCCL_OK;
}
