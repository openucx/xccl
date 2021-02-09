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
static xccl_status_t populate_mkey(xccl_mhba_team_t *team, int mem_access_flags,
                                   struct mlx5dv_mkey *mkey, void *mkey_entries,
                                   int repeat_count, int strided)
{
    xccl_status_t status;
    xccl_mhba_node_t *node = &team->node;
    ibv_wr_start(node->umr_qpx);
    node->umr_qpx->wr_id = 1; // First (and only) WR
    if (strided) {
        mlx5dv_wr_mr_interleaved(node->umr_mlx5dv_qp_ex, mkey, mem_access_flags,
                                 repeat_count, node->sbgp->group_size,
                                 (struct mlx5dv_mr_interleaved *)mkey_entries);
        xccl_mhba_debug("Execute the UMR WQE for populating the send/recv "
                        "MasterMKey lkey 0x%x",
                        mkey->lkey);
    } else {
        mlx5dv_wr_mr_list(node->umr_mlx5dv_qp_ex, mkey, mem_access_flags,
                          MAX_OUTSTANDING_OPS * team->max_num_of_columns, (struct ibv_sge *)mkey_entries);
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

static xccl_status_t create_and_populate_recv_team_mkey(xccl_mhba_team_t *team)
{
    xccl_status_t        status;
    xccl_mhba_node_t    *node = &team->node;
    int i, j;
    status = create_master_key(node, &node->team_recv_mkey, MAX_OUTSTANDING_OPS * team->max_num_of_columns);
    if (status != XCCL_OK) {
        return status;
    }
    struct ibv_sge *team_mkey_klm_entries =
        (struct ibv_sge *)calloc(MAX_OUTSTANDING_OPS * team->max_num_of_columns, sizeof(struct ibv_sge));
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for (j = 0; j < team->max_num_of_columns; j++) {
            team_mkey_klm_entries[(i * team->max_num_of_columns) + j].addr = 0;
            //length could be minimized for all mkeys beside the first, but no need because address space is big enough
            team_mkey_klm_entries[(i * team->max_num_of_columns) + j].length =
                    node->sbgp->group_size * team->max_msg_size * team->size;
            //todo check lkey or rkey
            team_mkey_klm_entries[(i * team->max_num_of_columns) + j].lkey =
                    node->ops[i].recv_mkeys[j]->rkey;
        }
    }
    status = populate_mkey(
        team, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        node->team_recv_mkey, team_mkey_klm_entries, 0, 0);
    if (status != XCCL_OK) {
        xccl_mhba_error("Failed to populate team mkey");
        if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
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
    int               i, j;
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        node->ops[i].send_mkeys = (struct mlx5dv_mkey **) malloc(sizeof(struct mlx5dv_mkey *) * team->max_num_of_columns);
        if(!node->ops[i].send_mkeys){
            xccl_mhba_error("Failed to malloc");
            xccl_mhba_destroy_mkeys(team, 1);
            return XCCL_ERR_NO_MEMORY;
        }
        node->ops[i].recv_mkeys = (struct mlx5dv_mkey **) malloc(sizeof(struct mlx5dv_mkey *) * team->max_num_of_columns);
        if(!node->ops[i].recv_mkeys){
            xccl_mhba_error("Failed to malloc");
            xccl_mhba_destroy_mkeys(team, 1);
            return XCCL_ERR_NO_MEMORY;
        }
        for(j=0;j<team->max_num_of_columns;j++) {
            status = create_master_key(node, &node->ops[i].send_mkeys[j],
                                       node->sbgp->group_size + 1);
            if (status != XCCL_OK) {
                xccl_mhba_error("create send masterkey[%d,%d] failed", i, j);
                xccl_mhba_destroy_mkeys(team, 1);
                return status;
            }
            status = create_master_key(node, &node->ops[i].recv_mkeys[j],
                                       node->sbgp->group_size + 1);
            if (status != XCCL_OK) {
                xccl_mhba_error("create recv masterkey[%d,%d] failed", i, j);
                xccl_mhba_destroy_mkeys(team, 1);
                return status;
            }
        }
    }
    status = create_and_populate_recv_team_mkey(team);
    if (status != XCCL_OK) {
        xccl_mhba_error("create recv top masterkey failed");
        xccl_mhba_destroy_mkeys(team, 1);
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
    xccl_mhba_node_t *node                  = &team->node;
    int               recv_mem_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    int               i;
    xccl_status_t     status;
    int repeat_count = req->num_of_blocks_columns ? team->net.sbgp->group_size : team->size/req->block_size;
    int n_mkeys = req->num_of_blocks_columns ? req->num_of_blocks_columns : 1;
    if(xccl_mhba_get_my_ctrl(team, req->seq_index)->mkey_cache_flag & XCCL_MHBA_NEED_SEND_MKEY_UPDATE) {
        for (i = 0; i < n_mkeys;i++) {
            status = populate_mkey(
                    team, send_mem_access_flags, node->ops[req->seq_index].send_mkeys[i],
                    node->ops[req->seq_index].send_umr_data[i], repeat_count, 1);
            if (status != XCCL_OK) {
                xccl_mhba_error("Failed to populate send umr[%d,%d]", req->seq_index, i);
                return status;
            }
        }
    }
    if(xccl_mhba_get_my_ctrl(team, req->seq_index)->mkey_cache_flag & XCCL_MHBA_NEED_RECV_MKEY_UPDATE) {
        for (i = 0; i < n_mkeys;i++) {
            status = populate_mkey(
                    team, recv_mem_access_flags, node->ops[req->seq_index].recv_mkeys[i],
                    node->ops[req->seq_index].recv_umr_data[i], repeat_count, 1);
            if (status != XCCL_OK) {
                xccl_mhba_error("Failed to populate recv umr[%d,%d]", req->seq_index, i);
                return status;
            }
        }
    }
    return XCCL_OK;
}

static void update_mkey_entry(xccl_mhba_node_t *node, xccl_mhba_coll_req_t *req,
                              int direction_send)
{
    struct mlx5dv_mr_interleaved *mkey_entry;
    struct ibv_mr *buff = direction_send ? req->send_rcache_region_p->mr : req->recv_rcache_region_p->mr;
    int i;
    if (!req->num_of_blocks_columns) {
        mkey_entry =
                (struct mlx5dv_mr_interleaved *) (direction_send ?
                                                  node->ops[req->seq_index].my_send_umr_data[0]
                                                                 : node->ops[req->seq_index].my_recv_umr_data[0]);
        mkey_entry->addr = (uintptr_t) buff->addr;
        mkey_entry->bytes_count = req->block_size * req->args.buffer_info.len;
        mkey_entry->bytes_skip = 0;
        mkey_entry->lkey = direction_send ? buff->lkey : buff->rkey;
        xccl_mhba_debug("%s MasterMKey Strided KLM entries[%d]: addr = 0x%x, "
                        "bytes_count = %d, bytes_skip = %d,lkey=0x%x",
                        direction_send ? "send" : "recv", node->sbgp->group_rank,
                        mkey_entry->addr, mkey_entry->bytes_count,
                        mkey_entry->bytes_skip, mkey_entry->lkey);
    } else {
        for(i=0; i< req->num_of_blocks_columns;i++){
            mkey_entry =
                    (struct mlx5dv_mr_interleaved *) (direction_send ?
                                                      node->ops[req->seq_index].my_send_umr_data[i] :
                                                      node->ops[req->seq_index].my_recv_umr_data[i]);
            mkey_entry->addr = (uintptr_t) buff->addr + i * (req->block_size * req->args.buffer_info.len);
            mkey_entry->bytes_count = (i == (req->num_of_blocks_columns - 1)) ? ((node->sbgp->group_size % req->block_size)
                    * req->args.buffer_info.len) : (req->block_size * req->args.buffer_info.len);
            mkey_entry->bytes_skip = (i == (req->num_of_blocks_columns - 1)) ?
                    ((node->sbgp->group_size - (node->sbgp->group_size % req->block_size)) * req->args.buffer_info.len) :
                    ((node->sbgp->group_size - req->block_size) * req->args.buffer_info.len);
            mkey_entry->lkey = direction_send ? buff->lkey : buff->rkey;
            xccl_mhba_debug("%s MasterMKey Strided KLM entries[%d,%d]: addr = 0x%x, "
                            "bytes_count = %d, bytes_skip = %d,lkey=0x%x",
                            direction_send ? "send" : "recv", node->sbgp->group_rank, i,
                            mkey_entry->addr, mkey_entry->bytes_count,
                            mkey_entry->bytes_skip, mkey_entry->lkey);
        }
    }
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
xccl_status_t xccl_mhba_destroy_mkeys(xccl_mhba_team_t *team, int error_mode)
{
    int           i, j;
    xccl_mhba_node_t *node = &team->node;
    xccl_status_t status = XCCL_OK;
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        for(j=0;j<team->max_num_of_columns;j++) {
            if (mlx5dv_destroy_mkey(node->ops[i].send_mkeys[j])) {
                if (!error_mode) {
                    xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                    status = XCCL_ERR_NO_MESSAGE;
                }
            }
            if (mlx5dv_destroy_mkey(node->ops[i].recv_mkeys[j])) {
                if (!error_mode) {
                    xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                    status = XCCL_ERR_NO_MESSAGE;
                }
            }
        }
       free(node->ops[i].send_mkeys);
       free(node->ops[i].recv_mkeys);
    }
    if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
        if (!error_mode) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
            status = XCCL_ERR_NO_MESSAGE;
        }
    }
    return status;
}
