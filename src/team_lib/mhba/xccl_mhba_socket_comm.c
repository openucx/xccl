/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"

xccl_status_t send_data(int root, int ctx_fd, uint32_t pd_handle, int *shared_ctx_fd_ptr, uint32_t
                                *shared_pd_handle_ptr){
    return XCCL_OK;

}

xccl_status_t xccl_mhba_share_ctx_pd(int root, xccl_mhba_node_t *node, int ctx_fd, uint32_t pd_handle,
                                     xccl_mhba_context_t *ctx){
    int      shared_ctx_fd;
    uint32_t shared_pd_handle;
    xccl_status_t status = send_data(root, ctx_fd, pd_handle, &shared_ctx_fd, &shared_pd_handle);
    if (XCCL_OK != status){
        return status;
    }
    if (root != node->sbgp->group_rank) {
        node->shared_ctx = ibv_import_device(shared_ctx_fd);
        if (!node->shared_ctx) {
            xccl_mhba_error("Import context failed");
            return XCCL_ERR_NO_MESSAGE;
        }
        node->shared_pd = ibv_import_pd(node->shared_ctx, shared_pd_handle);
        if (!node->shared_pd) {
            xccl_mhba_error("Import PD failed");
            if(ibv_close_device(node->shared_ctx)){
                xccl_mhba_error("imported context close failed");
            }
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    else{
        node->shared_ctx = ctx->ib_ctx;
        node->shared_pd  = ctx->ib_pd;
    }
    return XCCL_OK;
}

xccl_status_t xccl_mhba_remove_shared_ctx_pd(int root, xccl_mhba_node_t *node){
    if (root != node->sbgp->group_rank) {
        ibv_unimport_pd(node->shared_pd);
        if(ibv_close_device(node->shared_ctx)){
            xccl_mhba_error("imported context close failed");
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    return XCCL_OK;
}
