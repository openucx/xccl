/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"
#include "xccl_mhba_collective.h"
#include "utils/utils.h"

static xccl_status_t create_umr_qp(xccl_mhba_node_t *node)
{
    struct ibv_qp_init_attr_ex umr_init_attr_ex;
    struct mlx5dv_qp_init_attr umr_mlx5dv_qp_attr;

    xccl_mhba_debug("Create UMR QP (and CQ)");
    node->umr_cq = ibv_create_cq(node->shared_ctx, UMR_CQ_SIZE, NULL, NULL, 0);
    if (node->umr_cq == NULL) {
        xccl_mhba_error("UMR CQ creation failed");
        goto umr_cq_creation_failed;
    }

    memset(&umr_mlx5dv_qp_attr, 0, sizeof(umr_mlx5dv_qp_attr));
    memset(&umr_init_attr_ex, 0, sizeof(umr_init_attr_ex));

    umr_mlx5dv_qp_attr.comp_mask =
        MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_mlx5dv_qp_attr.create_flags = 0;
    umr_mlx5dv_qp_attr.send_ops_flags =
        MLX5DV_QP_EX_WITH_MR_LIST | MLX5DV_QP_EX_WITH_MR_INTERLEAVED;

    umr_init_attr_ex.send_cq          = node->umr_cq;
    umr_init_attr_ex.recv_cq          = node->umr_cq;
    umr_init_attr_ex.cap.max_send_wr  = 1;
    umr_init_attr_ex.cap.max_recv_wr  = 1;
    umr_init_attr_ex.cap.max_send_sge = 1;
    umr_init_attr_ex.cap.max_recv_sge = 1;
    // `max_inline_data` determines the WQE size that the QP will support.
    // The 'max_inline_data' should be modified only when the number of
    // arrays to interleave is greater than 3.
    //TODO query the devices what is max supported
    umr_init_attr_ex.cap.max_inline_data =
        828; // the max number possible, Sergey Gorenko's email
    umr_init_attr_ex.qp_type = IBV_QPT_RC;
    umr_init_attr_ex.comp_mask =
        IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_init_attr_ex.pd = node->shared_pd;
    umr_init_attr_ex.send_ops_flags |= IBV_QP_EX_WITH_SEND;
    node->umr_qp = mlx5dv_create_qp(node->shared_ctx, &umr_init_attr_ex,
                                    &umr_mlx5dv_qp_attr);
    if (node->umr_qp == NULL) {
        xccl_mhba_error("UMR QP (qp) creation failed");
        goto umr_qp_creation_failed;
    }
    xccl_mhba_debug("UMR QP created. Returned with cap.max_inline_data = %d",
                    umr_init_attr_ex.cap.max_inline_data);

    node->umr_qpx = ibv_qp_to_qp_ex(node->umr_qp);
    if (node->umr_qpx == NULL) {
        xccl_mhba_error("UMR qp_ex creation failed");
        goto umr_qpx_creation_failed;
    }

    // Turning on the IBV_SEND_SIGNALED option, will cause the reported work comletion to be with MLX5DV_WC_UMR opcode.
    // The option IBV_SEND_INLINE is required by the current API.
    node->umr_qpx->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    node->umr_mlx5dv_qp_ex  = mlx5dv_qp_ex_from_ibv_qp_ex(node->umr_qpx);
    if (node->umr_mlx5dv_qp_ex == NULL) {
        xccl_mhba_error("UMR qp_ex (mlx5dv_qp) creation failed");
        goto umr_mlx5dv_qp_creation_failed;
    }
    return XCCL_OK;

umr_mlx5dv_qp_creation_failed:
umr_qpx_creation_failed:
    if (ibv_destroy_qp(node->umr_qp)) {
        xccl_mhba_error("UMR qp destroy failed (errno=%d)", errno);
    }
umr_qp_creation_failed:
    if (ibv_destroy_cq(node->umr_cq)) {
        xccl_mhba_error("UMR cq destroy failed (errno=%d)", errno);
    }
umr_cq_creation_failed:
    return XCCL_ERR_NO_MESSAGE;
}

/**
 * Create and connect UMR qp & cq.
 * @param ctx mhba team context
 * @param node struct of the current process's node
 */
xccl_status_t xccl_mhba_init_umr(xccl_mhba_context_t *ctx,
                                 xccl_mhba_node_t    *node)
{
    struct ibv_port_attr port_attr;
    xccl_status_t        status;
    status = create_umr_qp(node);
    if (status != XCCL_OK) {
        return status;
    }
    if (ibv_query_port(node->umr_qp->context, ctx->ib_port, &port_attr)) {
        xccl_mhba_error("Couldn't get port info (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    status = xccl_mhba_qp_connect(node->umr_qp, node->umr_qp->qp_num,
                                  port_attr.lid, ctx->ib_port);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

static xccl_status_t create_master_key(xccl_mhba_node_t    *node,
                                       struct mlx5dv_mkey **mkey_ptr,
                                       int                  num_of_entries)
{
    struct mlx5dv_mkey          *mkey;
    struct mlx5dv_mkey_init_attr umr_mkey_init_attr;
    memset(&umr_mkey_init_attr, 0, sizeof(umr_mkey_init_attr));
    umr_mkey_init_attr.pd           = node->shared_pd;
    umr_mkey_init_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    // Defines how many entries the Mkey will support.
    // In case the MKey is used as "strided-KLM based MKey", the number
    // of entries that is needed is increased by one because one entry is
    // consumed by the "strided header" (see mlx5dv_wr_post manual).
    umr_mkey_init_attr.max_entries = num_of_entries;
    mkey                           = mlx5dv_create_mkey(&umr_mkey_init_attr);
    if (mkey == NULL) {
        xccl_mhba_error("MasterMKey creation failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    xccl_mhba_debug("umr_master_key_dv_mkey: lkey=0x%x, with %d entries",
                    mkey->lkey, num_of_entries);
    *mkey_ptr = mkey;
    return XCCL_OK;
}

static xccl_status_t poll_umr_cq(xccl_mhba_node_t *node)
{
    struct ibv_wc wc;
    int           ret = 0;
    while (!ret) {
        ret = ibv_poll_cq(node->umr_cq, 1, &wc);
        if (ret < 0) {
            xccl_mhba_error("ibv_poll_cq() failed for UMR execution");
            return XCCL_ERR_NO_MESSAGE;
        } else if (ret > 0) {
            if (wc.status != IBV_WC_SUCCESS || wc.opcode != MLX5DV_WC_UMR) {
                xccl_mhba_error("umr cq returned incorrect completion: status "
                                "%s, opcode %d",
                                ibv_wc_status_str(wc.status), wc.opcode);
                return XCCL_ERR_NO_MESSAGE;
            }
        }
    }
    xccl_mhba_debug("Successfully executed the UMR WQE");
    return XCCL_OK;
}

// Execute the UMR WQE for populating the UMR's MasterMKey
static xccl_status_t populate_mkey(xccl_mhba_node_t *node, int mem_access_flags,
                                   struct mlx5dv_mkey *mkey, void *mkey_entries,
                                   int block_size, int team_size, int strided)
{
    xccl_status_t status;
    ibv_wr_start(node->umr_qpx);
    node->umr_qpx->wr_id = 1; // First (and only) WR
    if (strided) {
        int repeat_count = xccl_round_up(team_size, block_size);
        mlx5dv_wr_mr_interleaved(node->umr_mlx5dv_qp_ex, mkey, mem_access_flags,
                                 repeat_count, node->sbgp->group_size,
                                 (struct mlx5dv_mr_interleaved *)mkey_entries);
        xccl_mhba_debug("Execute the UMR WQE for populating the send/recv "
                        "MasterMKey lkey 0x%x",
                        mkey->lkey);
    } else {
        mlx5dv_wr_mr_list(node->umr_mlx5dv_qp_ex, mkey, mem_access_flags,
                          MAX_OUTSTANDING_OPS, (struct ibv_sge *)mkey_entries);
        xccl_mhba_debug(
            "Execute the UMR WQE for populating the team MasterMKeys lkey 0x%x",
            mkey->lkey);
    }
    if (ibv_wr_complete(node->umr_qpx)) {
        xccl_mhba_error("UMR WQE failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    status = poll_umr_cq(node);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

static xccl_status_t create_and_populate_team_mkey(xccl_mhba_team_t *team,
                                                   int               send)
{
    xccl_status_t        status;
    xccl_mhba_node_t    *node = &team->node;
    struct mlx5dv_mkey **mkey =
        send ? &node->team_send_mkey : &node->team_recv_mkey;
    int i;
    status = create_master_key(node, mkey, MAX_OUTSTANDING_OPS);
    if (status != XCCL_OK) {
        return status;
    }
    struct ibv_sge *team_mkey_klm_entries =
        (struct ibv_sge *)calloc(MAX_OUTSTANDING_OPS, sizeof(struct ibv_sge));
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        team_mkey_klm_entries[i].addr = 0;
        team_mkey_klm_entries[i].length =
            node->sbgp->group_size * team->max_msg_size * team->size;
        //todo check lkey or rkey
        team_mkey_klm_entries[i].lkey =
            send ? node->ops[i].send_mkey->lkey : node->ops[i].recv_mkey->rkey;
    }
    status = populate_mkey(
        node, send ? 0 : IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        *mkey, team_mkey_klm_entries, 0, 0, 0);
    if (status != XCCL_OK) {
        xccl_mhba_error("Failed to populate team mkey");
        if (mlx5dv_destroy_mkey(*mkey)) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
        }
        return status;
    }
    free(team_mkey_klm_entries);
    return XCCL_OK;
}

/**
 * Create mkeys for all outstanding AlltoAll ops in each rank. Creats team mkey, and execute the team mkey's
 * population WQE
 * @param node struct of the current process's node
 * @param team_size number of processes in team
 */
xccl_status_t xccl_mhba_init_mkeys(xccl_mhba_team_t *team)
{
    xccl_status_t     status;
    xccl_mhba_node_t *node = &team->node;
    int               i;
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        status = create_master_key(node, &node->ops[i].send_mkey,
                                   node->sbgp->group_size + 1);
        if (status != XCCL_OK) {
            xccl_mhba_error("create send masterkey[%d] failed", i);
            xccl_mhba_destroy_mkeys(node, 1);
            return status;
        }
        status = create_master_key(node, &node->ops[i].recv_mkey,
                                   node->sbgp->group_size + 1);
        if (status != XCCL_OK) {
            xccl_mhba_error("create recv masterkey[%d] failed", i);
            xccl_mhba_destroy_mkeys(node, 1);
            return status;
        }
    }
    status = create_and_populate_team_mkey(team, 1);
    if (status != XCCL_OK) {
        xccl_mhba_error("create send top masterkey failed");
        xccl_mhba_destroy_mkeys(node, 1);
        return status;
    }
    status = create_and_populate_team_mkey(team, 0);
    if (status != XCCL_OK) {
        xccl_mhba_error("create recv top masterkey failed");
        xccl_mhba_destroy_mkeys(node, 1);
        return status;
    }
    return XCCL_OK;
}

/**
 * Execute UMR WQE to populate mkey of specific AlltoAll operation, after the mkey entries were already updated
 * @param team struct of the current team
 * @param req current AlltoAll operation request
 */
xccl_status_t xccl_mhba_populate_send_recv_mkeys(xccl_mhba_team_t     *team,
                                                 xccl_mhba_coll_req_t *req)
{
    int               send_mem_access_flags = 0;
    int               index                 = SEQ_INDEX(req->seq_num);
    xccl_mhba_node_t *node                  = &team->node;
    int               recv_mem_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    xccl_status_t     status;
    if(xccl_mhba_get_my_ctrl(team, index)->mkey_cache_flag & XCCL_MHBA_NEED_SEND_MKEY_UPDATE) {
        status = populate_mkey(
                node, send_mem_access_flags, node->ops[index].send_mkey,
                node->ops[index].send_umr_data, req->block_size, team->size, 1);
        if (status != XCCL_OK) {
            xccl_mhba_error("Failed to populate send umr[%d]", index);
            return status;
        }
    }
    if(xccl_mhba_get_my_ctrl(team, index)->mkey_cache_flag & XCCL_MHBA_NEED_RECV_MKEY_UPDATE) {
        status = populate_mkey(
                node, recv_mem_access_flags, node->ops[index].recv_mkey,
                node->ops[index].recv_umr_data, req->block_size, team->size, 1);
        if (status != XCCL_OK) {
            xccl_mhba_error("Failed to populate recv umr[%d]", index);
            return status;
        }
    }
    return XCCL_OK;
}

static void update_mkey_entry(xccl_mhba_node_t *node, xccl_mhba_coll_req_t *req,
                              int direction_send)
{
    int                           index = SEQ_INDEX(req->seq_num);
    struct mlx5dv_mr_interleaved *mkey_entry =
        (struct mlx5dv_mr_interleaved *)(direction_send ?
                                         node->ops[index].my_send_umr_data
                                         : node->ops[index].my_recv_umr_data);
    struct ibv_mr *buff = direction_send ? req->send_rcache_region_p->mr : req->recv_rcache_region_p->mr;
    mkey_entry->addr    = (uintptr_t)buff->addr;
    mkey_entry->bytes_count = req->block_size * req->args.buffer_info.len;
    mkey_entry->bytes_skip  = 0;
    mkey_entry->lkey        = direction_send ? buff->lkey : buff->rkey;
    xccl_mhba_debug("%s MasterMKey Strided KLM entries[%d]: addr = 0x%x, "
                    "bytes_count = %d, bytes_skip = %d,lkey=0x%x",
                    direction_send ? "send" : "recv", node->sbgp->group_rank,
                    mkey_entry->addr, mkey_entry->bytes_count,
                    mkey_entry->bytes_skip, mkey_entry->lkey);
}

/**
 * Update the UMR klm entry (ranks send & receive buffers) for specific AlltoAll operation
 * @param node struct of the current process's node
 * @param req AlltoAll operation request object
 */
xccl_status_t xccl_mhba_update_mkeys_entries(xccl_mhba_node_t     *node,
                                             xccl_mhba_coll_req_t *req)
{
    update_mkey_entry(node, req, 1);
    update_mkey_entry(node, req, 0);
    return XCCL_OK;
}

/**
 * Clean UMR qp & cq
 * @param node struct of the current process's node
 */
xccl_status_t xccl_mhba_destroy_umr(xccl_mhba_node_t *node)
{
    if (ibv_destroy_qp(node->umr_qp)) {
        xccl_mhba_error("umr qp destroy failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    if (ibv_destroy_cq(node->umr_cq)) {
        xccl_mhba_error("umr cq destroy failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

/**
 * Clean all mkeys -  operation mkeys, team mkeys
 * @param node struct of the current process's node
 * @param error_mode boolean - ordinary destroy or destroy due to an earlier error
 */
xccl_status_t xccl_mhba_destroy_mkeys(xccl_mhba_node_t *node, int error_mode)
{
    int           i;
    xccl_status_t status = XCCL_OK;
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        if (mlx5dv_destroy_mkey(node->ops[i].send_mkey)) {
            if (!error_mode) {
                xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                status = XCCL_ERR_NO_MESSAGE;
            }
        }
        if (mlx5dv_destroy_mkey(node->ops[i].recv_mkey)) {
            if (!error_mode) {
                xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                status = XCCL_ERR_NO_MESSAGE;
            }
        }
    }
    if (mlx5dv_destroy_mkey(node->team_send_mkey)) {
        if (!error_mode) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
            status = XCCL_ERR_NO_MESSAGE;
        }
    }
    if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
        if (!error_mode) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
            status = XCCL_ERR_NO_MESSAGE;
        }
    }
    return status;
}
