/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"
#include "xccl_mhba_collective.h"


static xccl_status_t poll_cq(struct ibv_cq *cq, int expected_wc_num, struct ibv_wc *actual_wc) {
    int total_received_completions = 0;
    int new_received_completions = 0;
    while (expected_wc_num > 0) {
        new_received_completions = ibv_poll_cq(cq, expected_wc_num, actual_wc);
        if (new_received_completions < 0) {
            xccl_mhba_error("ibv_poll_cq() failed for UMR execution");
            return XCCL_ERR_NO_MESSAGE;
        }
        total_received_completions += new_received_completions;
        expected_wc_num -= new_received_completions;
    }
    return XCCL_OK;
}

static xccl_status_t create_umr_qp(xccl_mhba_node_t *node) {
    struct ibv_qp_init_attr_ex umr_init_attr_ex;
    struct mlx5dv_qp_init_attr umr_mlx5dv_qp_attr;

    xccl_mhba_debug("Create UMR QP (and CQ)");
    node->umr_cq = ibv_create_cq(node->shared_ctx, UMR_CQ_SIZE, NULL, NULL, 0);
    if (node->umr_cq == NULL) {
        xccl_mhba_error("UMR CQ creation failed");
        goto umr_cq_creation_failed;
    }

    memset(&umr_mlx5dv_qp_attr, 0, sizeof(umr_mlx5dv_qp_attr));
    umr_mlx5dv_qp_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD; //todo ask Rami
    umr_mlx5dv_qp_attr.create_flags = 0;
    memset(&umr_mlx5dv_qp_attr.dc_init_attr,0,sizeof(umr_mlx5dv_qp_attr.dc_init_attr)); //todo: check is right
    umr_mlx5dv_qp_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MR_LIST | MLX5DV_QP_EX_WITH_MR_INTERLEAVED;

    memset(&umr_init_attr_ex, 0, sizeof(umr_init_attr_ex));
    umr_init_attr_ex.send_cq = node->umr_cq;
    umr_init_attr_ex.recv_cq = node->umr_cq;
    umr_init_attr_ex.cap.max_send_wr = 1;
    umr_init_attr_ex.cap.max_recv_wr = 0;
    umr_init_attr_ex.cap.max_send_sge = 1;
    umr_init_attr_ex.cap.max_recv_sge = 0;
    // `max_inline_data` determines the WQE size that the QP will support.
    // The 'max_inline_data' should be modified only when the number of
    // arrays to interleave is greater than 3.
    umr_init_attr_ex.cap.max_inline_data = 828; // the max number possible, Sergey Gorenko's email
    umr_init_attr_ex.qp_type = IBV_QPT_RC;
    umr_init_attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_init_attr_ex.pd = node->shared_pd;
    umr_init_attr_ex.send_ops_flags |= IBV_QP_EX_WITH_SEND;
    node->umr_qp = mlx5dv_create_qp(node->shared_ctx, &umr_init_attr_ex, &umr_mlx5dv_qp_attr);
    if (node->umr_qp == NULL) {
        xccl_mhba_error("UMR QP (qp) creation failed");
        goto umr_qp_creation_failed;
    }
    xccl_mhba_debug("UMR QP created. Returned with cap.max_inline_data = %d", umr_init_attr_ex.cap.max_inline_data);

    node->umr_qpx = ibv_qp_to_qp_ex(node->umr_qp);
    if (node->umr_qpx == NULL) {
        xccl_mhba_error("UMR qp_ex creation failed");
        goto umr_qpx_creation_failed;
    }

    // Turning on the IBV_SEND_SIGNALED option, will cause the reported work comletion to be with MLX5DV_WC_UMR opcode.
    // The option IBV_SEND_INLINE is required by the current API.
    node->umr_qpx->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

    node->umr_mlx5dv_qp_ex = mlx5dv_qp_ex_from_ibv_qp_ex(node->umr_qpx);
    if (node->umr_mlx5dv_qp_ex == NULL) {
        xccl_mhba_error("UMR qp_ex (mlx5dv_qp) creation failed");
        goto umr_mlx5dv_qp_creation_failed;
    }
    return XCCL_OK;

    umr_mlx5dv_qp_creation_failed:
    umr_qpx_creation_failed:
    if(ibv_destroy_qp(node->umr_qp)){
        xccl_mhba_error("UMR qp destroy failed (errno=%d)", errno);
    }
    umr_qp_creation_failed:
    if(ibv_destroy_cq(node->umr_cq)){
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
xccl_status_t xccl_mhba_init_umr(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node) {
    struct ibv_port_attr port_attr;
    xccl_status_t status;
    status = create_umr_qp(node);
    if (status != XCCL_OK) {
        return status;
    }
    if (ibv_query_port(node->umr_qp->context, ctx->ib_port, &port_attr)) {
        xccl_mhba_error("Couldn't get port info (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    status = xccl_mhba_remote_qp_connect(node->umr_qp, node->umr_qp->qp_num, port_attr.lid, ctx->ib_port);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

static xccl_status_t
create_master_key(xccl_mhba_node_t *node, struct mlx5dv_mkey **mkey_ptr,int num_of_entries) {
    struct mlx5dv_mkey* mkey;
    struct mlx5dv_mkey_init_attr umr_mkey_init_attr;
    memset(&umr_mkey_init_attr, 0, sizeof(umr_mkey_init_attr));
    umr_mkey_init_attr.pd = node->shared_pd;
    umr_mkey_init_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    // Defines how many entries the Mkey will support.
    // In case the MKey is used as "strided-KLM based MKey", the number
    // of entries that is needed is increased by one because one entry is
    // consumed by the "strided header" (see mlx5dv_wr_post manual).
    umr_mkey_init_attr.max_entries = num_of_entries;
    mkey = mlx5dv_create_mkey(&umr_mkey_init_attr);
    if (mkey == NULL) {
        xccl_mhba_error("MasterMKey creation failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    xccl_mhba_debug("umr_master_key_dv_mkey: lkey=0x%x, with %d entries", mkey->lkey, num_of_entries);
    *mkey_ptr = mkey;
    return XCCL_OK;
}

static xccl_status_t poll_umr_cq(xccl_mhba_node_t *node) {
    const int expected_completions_num = 1;
    struct ibv_wc work_completion[expected_completions_num];
    memset(work_completion, 0, expected_completions_num * sizeof(struct ibv_wc));
    int num_completions;
    xccl_status_t status = poll_cq(node->umr_cq, expected_completions_num, work_completion);
    if (status != XCCL_OK) {
        xccl_mhba_error("UMR WQE on UMR QP failed");
        return status;
    }
    if (work_completion[0].status != IBV_WC_SUCCESS) {
        xccl_mhba_error("UMR CQ returned completion with status %s (%d)",
                        ibv_wc_status_str(work_completion[0].status), work_completion[0].status);
        return XCCL_ERR_NO_MESSAGE;
    }
    if (work_completion[0].opcode != MLX5DV_WC_UMR) {
        xccl_mhba_error("Got unexpected completion opcode for the UMR QP: %d", work_completion[0].opcode);
        return XCCL_ERR_NO_MESSAGE;
    }
    xccl_mhba_debug("Successfully executed the UMR WQE");
    return XCCL_OK;
}

// Execute the UMR WQE for populating the UMR's MasterMKey
static xccl_status_t
populate_mkey(xccl_mhba_node_t *node, int mem_access_flags, struct mlx5dv_mkey *mkey,
              void *mkey_entries, int block_size, int strided, int num_of_entries) {
    xccl_status_t status;
    ibv_wr_start(node->umr_qpx);
    node->umr_qpx->wr_id = 1; // First (and only) WR
    if (strided) {
        int repeat_count = node->sbgp->group_size / block_size; //todo change per node PPN
        mlx5dv_wr_mr_interleaved(node->umr_mlx5dv_qp_ex, mkey, mem_access_flags,
                                 repeat_count, node->sbgp->group_size,
                                 (struct mlx5dv_mr_interleaved *) mkey_entries);
        xccl_mhba_debug("Execute the UMR WQE for populating the send/recv MasterMKey lkey 0x%x",mkey->lkey);
    } else{
        mlx5dv_wr_mr_list(node->umr_mlx5dv_qp_ex, mkey, mem_access_flags, num_of_entries, (struct
        ibv_sge*)mkey_entries);
        xccl_mhba_debug("Execute the UMR WQE for populating the team MasterMKeys lkey 0x%x",mkey->lkey);
    }
    if (ibv_wr_complete(node->umr_qpx)){
        xccl_mhba_error("UMR WQE failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    status = poll_umr_cq(node);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

static xccl_status_t create_and_populate_klm_mkey(xccl_mhba_node_t *node, int length, struct mlx5dv_mkey** mkey ,
        uint32_t* lkeys[], int send,int num_of_entries){
    xccl_status_t status;
    int i;
    status = create_master_key(node, mkey, num_of_entries);
    if (status != XCCL_OK) {
        return status;
    }
    struct ibv_sge* mkey_klm_entries = (struct ibv_sge*)calloc(num_of_entries, sizeof(struct ibv_sge));
    for (i = 0; i < num_of_entries; i++) {
        mkey_klm_entries[i].addr = 0;
        mkey_klm_entries[i].length = length;
        mkey_klm_entries[i].lkey = *lkeys[i];
    }
    status = populate_mkey(node,send ? 0 : IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,*mkey,mkey_klm_entries,0,
                           0,num_of_entries);
    if (status != XCCL_OK) {
        xccl_mhba_error("Failed to populate team mkey");
        if (mlx5dv_destroy_mkey(*mkey)) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
        }
        return status;
    }
    free(mkey_klm_entries);
    return XCCL_OK;
}

/**
 * Create mkeys for all outstanding AlltoAll operations in each rank. Creats team mkey, and execute the team mkey's
 * population WQE
 * @param node struct of the current process's node
 * @param team_size number of processes in team
 */
xccl_status_t xccl_mhba_init_mkeys(xccl_mhba_node_t *node, int net_size, int team_size) {
    xccl_status_t status;
    int i,j;
    uint32_t** send_node_lkeys = (uint32_t**) malloc(sizeof(uint32_t*)*net_size);
    uint32_t** recv_node_lkeys = (uint32_t**) malloc(sizeof(uint32_t*)*net_size);
    uint32_t** send_operation_lkeys = (uint32_t**) malloc(sizeof(uint32_t*)*MAX_CONCURRENT_OUTSTANDING_ALL2ALL);
    uint32_t** recv_operation_lkeys = (uint32_t**) malloc(sizeof(uint32_t*)*MAX_CONCURRENT_OUTSTANDING_ALL2ALL);
    for(i=0;i<MAX_CONCURRENT_OUTSTANDING_ALL2ALL;i++) {
        for(j=0;j<net_size;j++){
            status = create_master_key(node, &node->operations[i].nodes_umr[j].node_send_mkey,
                                                                            node->sbgp->group_size + 1);
            if (status != XCCL_OK) {
                xccl_mhba_error("create send node masterkey[%d,%d] failed",i,j);
                goto fail;
            }

            send_node_lkeys[j] = &node->operations[i].nodes_umr[j].node_send_mkey->lkey;

            status = create_master_key(node, &node->operations[i].nodes_umr[j].node_recv_mkey,
                                                                            node->sbgp->group_size + 1);
            if (status != XCCL_OK) {
                xccl_mhba_error("create recv node masterkey[%d,%d] failed",i,j);
                goto fail;
            }

            recv_node_lkeys[j] = &node->operations[i].nodes_umr[j].node_recv_mkey->rkey;
        }

        //todo add length array instead of fixed according to node PPN
        status = create_and_populate_klm_mkey(node,node->sbgp->group_size*MAX_MSG_SIZE*node->sbgp->group_size,
                                              &node->operations[i].send_mkey,send_node_lkeys, 1, net_size);
        if (status != XCCL_OK) {
            xccl_mhba_error("create send masterkey[%d] failed",i);
            goto fail;
        }

        send_operation_lkeys[i] = &node->operations[i].send_mkey->lkey;

        //todo add length array instead of fixed according to node PPN
        status = create_and_populate_klm_mkey(node,node->sbgp->group_size*MAX_MSG_SIZE*node->sbgp->group_size,
                                              &node->operations[i].recv_mkey,recv_node_lkeys, 0, net_size);
        if (status != XCCL_OK) {
            xccl_mhba_error("create recv masterkey[%d,%d] failed",i);
            goto fail;
        }

        recv_operation_lkeys[i] = &node->operations[i].recv_mkey->rkey;

    }
    free(send_node_lkeys);
    free(recv_node_lkeys);

    status = create_and_populate_klm_mkey(node,node->sbgp->group_size*MAX_MSG_SIZE*team_size,&node->team_send_mkey,
                                          send_operation_lkeys, 1,MAX_CONCURRENT_OUTSTANDING_ALL2ALL);
    if (status != XCCL_OK) {
        xccl_mhba_error("create team send mkey failed",i);
        goto fail;
    }
    status = create_and_populate_klm_mkey(node,node->sbgp->group_size*MAX_MSG_SIZE*team_size,&node->team_recv_mkey,
                                          recv_operation_lkeys, 0, MAX_CONCURRENT_OUTSTANDING_ALL2ALL);
    if (status != XCCL_OK) {
        xccl_mhba_error("create team recv mkey failed",i);
        goto fail;
    }
    free(send_operation_lkeys);
    free(recv_operation_lkeys);
    return XCCL_OK;

fail:
    free(send_node_lkeys);
    free(recv_node_lkeys);
    free(send_operation_lkeys);
    free(recv_operation_lkeys);
    xccl_mhba_destroy_mkeys(node,net_size,1);
    return XCCL_ERR_NO_MESSAGE;
}

/**
 * Execute UMR WQE to populate mkey of specific AlltoAll operation, after the mkey entries were already updated
 * @param team mhba curent team struct
 * @param request current mhba request struct
 */
xccl_status_t xccl_mhba_populate_send_recv_mkeys(xccl_mhba_team_t *team,xccl_mhba_coll_req_t *request){
    xccl_status_t status;
    int send_mem_access_flags = 0, j;
    int index = seq_index(request->seq_num);
    for(j=0;j<team->net.sbgp->group_size;j++) {
        status = populate_mkey(&team->node, send_mem_access_flags, team->node.operations[index].nodes_umr[j].node_send_mkey,
                               team->node.operations[index].nodes_umr[j].send_umr_data, request->block_size, 1, 0);
        if (status != XCCL_OK) {
            xccl_mhba_error("Failed to populate send umr[%d,%d]", index, j);
            return status;
        }
        int recv_mem_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        status = populate_mkey(&team->node, recv_mem_access_flags, team->node.operations[index].nodes_umr[j].node_recv_mkey,
                               team->node.operations[index].nodes_umr[j].recv_umr_data, request->block_size, 1, 0);
        if (status != XCCL_OK) {
            xccl_mhba_error("Failed to populate recv umr[%d,%d]", index, j);
            return status;
        }
    }
    return XCCL_OK;
}

static void update_mkey_entry(xccl_mhba_node_t *node, xccl_mhba_coll_req_t *req, int direction_send,xccl_mhba_net_t*
                            net){
    int index = seq_index(req->seq_num);
    int j;
    for(j=0;j<net->sbgp->group_size;j++) {
        struct mlx5dv_mr_interleaved *mkey_entry = (struct mlx5dv_mr_interleaved *) (direction_send ?
                                                     node->operations[index].nodes_umr[j].my_send_umr_data :
                                                     node->operations[index].nodes_umr[j].my_recv_umr_data);
        struct ibv_mr *buff = direction_send ? req->send_bf_mr : req->receive_bf_mr;
        mkey_entry->addr = (uintptr_t) buff->addr + j*node->sbgp->group_size*req->args.buffer_info.len; // TODO: Check why conversion to uintptr_t is needed and if it's correct
        mkey_entry->bytes_count = req->block_size * req->args.buffer_info.len;
        mkey_entry->bytes_skip = 0;
        mkey_entry->lkey = direction_send ? buff->lkey : buff->rkey; //todo check if same lkey along the memory
        xccl_mhba_debug(
                "%s MasterMKey Strided KLM entries[%d,%d]: addr = 0x%x, bytes_count = %d, bytes_skip = %d,lkey=0x%x",
                direction_send ? "send" : "recv", j,node->sbgp->group_rank, mkey_entry->addr,
                mkey_entry->bytes_count, mkey_entry->bytes_skip, mkey_entry->lkey);
    }
}

/**
 * Update the UMR klm entry (ranks send & receive buffers) for specific AlltoAll operation
 * @param node struct of the current process's node
 * @param req AlltoAll operation request object
 */
xccl_status_t
xccl_mhba_update_mkeys_entries(xccl_mhba_node_t *node, xccl_mhba_coll_req_t *req,xccl_mhba_net_t* net) {
    update_mkey_entry(node, req, 1,net);
    update_mkey_entry(node, req, 0,net);
    return XCCL_OK;
}

/**
 * Clean UMR qp & cq
 * @param node struct of the current process's node
 */
xccl_status_t xccl_mhba_destroy_umr(xccl_mhba_node_t *node) {
    if(node->umr_cq == NULL){
        return XCCL_OK;
    }
    if(ibv_destroy_qp(node->umr_qp)){
        xccl_mhba_error("umr qp destroy failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    if (ibv_destroy_cq(node->umr_cq)){
        xccl_mhba_error("umr cq destroy failed (errno=%d)", errno);
                                return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

/**
 * Clean all mkeys -  operation mkeys, team mkeys
 * @param node struct of the current process's node
 */
xccl_status_t xccl_mhba_destroy_mkeys(xccl_mhba_node_t *node, int net_size, int error_mode) {
    int i,j;
    xccl_status_t status = XCCL_OK;
    for(i=0;i<MAX_CONCURRENT_OUTSTANDING_ALL2ALL;i++) {
        for(j=0;j<net_size;j++){
            if (mlx5dv_destroy_mkey(node->operations[i].nodes_umr[j].node_send_mkey)) {
                if(!error_mode) {
                    xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                    status = XCCL_ERR_NO_MESSAGE;
                }
            }
            if (mlx5dv_destroy_mkey(node->operations[i].nodes_umr[j].node_recv_mkey)) {
                if(!error_mode) {
                    xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                    status = XCCL_ERR_NO_MESSAGE;
                }
            }
        }

        if (mlx5dv_destroy_mkey(node->operations[i].send_mkey)) {
            if(!error_mode) {
                xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                status = XCCL_ERR_NO_MESSAGE;
            }
        }
        if (mlx5dv_destroy_mkey(node->operations[i].recv_mkey)) {
            if(!error_mode) {
                xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
                status = XCCL_ERR_NO_MESSAGE;
            }
        }
    }
    if (mlx5dv_destroy_mkey(node->team_send_mkey)) {
        if(!error_mode) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
            status = XCCL_ERR_NO_MESSAGE;
        }
    }
    if (mlx5dv_destroy_mkey(node->team_recv_mkey)) {
        if(!error_mode) {
            xccl_mhba_error("mkey destroy failed(errno=%d)", errno);
            status = XCCL_ERR_NO_MESSAGE;
        }
    }
    return status;
}


