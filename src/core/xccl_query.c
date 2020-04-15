/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <xccl_lib.h>
#include <xccl_team_lib.h>
#include <stdlib.h>

xccl_status_t xccl_get_tl_list(xccl_lib_h lib, xccl_tl_id_t **tls,
                               unsigned *tl_count)
{
    int i;

    *tl_count = lib->n_libs_opened;
    (*tls) = (xccl_tl_id_t*)malloc((*tl_count) * sizeof(xccl_tl_id_t));
    for(i = 0; i < *tl_count; i++) {
        (*tls)[i] = lib->libs[i]->id;
    }

    return XCCL_OK;
}

void xccl_free_tl_list(xccl_tl_id_t *tls)
{
    free(tls);
}

xccl_status_t xccl_tl_query(xccl_lib_h lib, xccl_tl_id_t *tl_id,
                            xccl_tl_attr_t *tl_attr)
{
    xccl_team_lib_t *tl = NULL;
    xccl_status_t   status;
    int             i;

    for(i = 0; i < lib->n_libs_opened; i++){
        if (lib->libs[i]->id == *tl_id) {
            tl = lib->libs[i];
            break;
        }
    }

    if (tl == NULL) {
        xccl_warn("Wrong tl_id specified (%d)", *tl_id);
        return XCCL_ERR_INVALID_PARAM;
    }

    if (tl->team_lib_query == NULL) {
        xccl_warn("TL %s is not supported", xccl_tl_str(*tl_id));
        return XCCL_ERR_NOT_IMPLEMENTED;
    }

    status = tl->team_lib_query(tl, tl_attr);
    
    return status;
}

void xccl_free_tl_attr(xccl_tl_attr_t *attr) {
    if (attr->field_mask & XCCL_TL_ATTR_FILED_DEVICES) {
        free(attr->devices);
    }
}
