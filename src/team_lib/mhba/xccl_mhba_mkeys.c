/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"


//todo extract to helper header
int poll_cq(struct ibv_cq *cq, int expected_wc_num, int *actual_wc_num, struct ibv_wc actual_wc[]) {
    *actual_wc_num = -1; // Initial value.
    long int poll_for_completion_retry = 10;
    int total_received_completions = 0;
    int new_received_completions = 0;
    int completions_to_wait_for = expected_wc_num;

    // TODO: Consider doing this instead of requiring the user to initialize
    // the array of completions.
    //memset((void*)actual_wc, 0, expected_wc_num*sizeof(struct ibv_wc));

    while ((completions_to_wait_for > 0) && (poll_for_completion_retry > 0)) {
        new_received_completions = ibv_poll_cq(cq, completions_to_wait_for, &actual_wc[total_received_completions]);
        if (new_received_completions < 0) {
            xccl_mhba_error("ibv_poll_cq() failed");
            return new_received_completions;
        }
        total_received_completions += new_received_completions;
        completions_to_wait_for -= new_received_completions;
        if (completions_to_wait_for > 0) {
            poll_for_completion_retry--;
            sleep(1); // todo ?
        }
    }

    if (total_received_completions == 0) {
        xccl_mhba_error("No CQEs were generated. Retry counter exceeded");
        return -1;
    }

    if (completions_to_wait_for > 0) {
        xccl_mhba_error("Not enough CQE were generated. Retry counter exceeded");
        return total_received_completions;
    }
    *actual_wc_num = total_received_completions;
    return total_received_completions;
}

xccl_status_t create_umr_qp(xccl_mhba_context_t *ctx) {
    struct ibv_qp_init_attr_ex umr_init_attr_ex;
    struct mlx5dv_qp_init_attr umr_mlx5dv_qp_attr;

    xccl_mhba_debug("Create UMR QP (and CQ)");
    ctx->umr_cq = ibv_create_cq(ctx->ib_ctx, UMR_CQ_SIZE, NULL, NULL, 0);
    if (ctx->umr_cq == NULL) {
        xccl_mhba_error("UMR CQ creation failed");
        goto umr_cq_creation_failed;
    }

    memset(&umr_mlx5dv_qp_attr, 0, sizeof(umr_mlx5dv_qp_attr));
    umr_mlx5dv_qp_attr.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS, IBV_QP_INIT_ATTR_PD;
    umr_mlx5dv_qp_attr.create_flags = 0;
    memset(&umr_mlx5dv_qp_attr.dc_init_attr,0,sizeof(umr_mlx5dv_qp_attr.dc_init_attr)); //todo: check is right
    umr_mlx5dv_qp_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MR_LIST | MLX5DV_QP_EX_WITH_MR_INTERLEAVED;

    memset(&umr_init_attr_ex, 0, sizeof(umr_init_attr_ex));
    umr_init_attr_ex.send_cq = ctx->umr_cq;
    umr_init_attr_ex.recv_cq = ctx->umr_cq;
    umr_init_attr_ex.cap.max_send_wr = 1;
    umr_init_attr_ex.cap.max_recv_wr = 0;
    umr_init_attr_ex.cap.max_send_sge = 1;
    umr_init_attr_ex.cap.max_recv_sge = 0;
    // `max_inline_data` determines the WQE size that the QP will support.
    // The 'max_inline_data' should be modified only when the number of
    // arrays to interleave is greater than 3.
    umr_init_attr_ex.cap.max_inline_data = 256; // todo: in final stage the UMR won't be a WQE but a raw request, in this stage we might hit a ceiling with that size, need to test-and-err. for big sbgp_size might be a problem, depend on number of UMR entries
    umr_init_attr_ex.qp_type = IBV_QPT_RC;
    umr_init_attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    umr_init_attr_ex.pd = ctx->ib_pd;
    umr_init_attr_ex.send_ops_flags |= IBV_QP_EX_WITH_SEND;
    ctx->umr_qp = mlx5dv_create_qp(ctx->ib_ctx, &umr_init_attr_ex, &umr_mlx5dv_qp_attr);
    if (ctx->umr_qp == NULL) {
        xccl_mhba_error("UMR QP (qp) creation failed");
        goto umr_qp_creation_failed;
    }
    xccl_mhba_debug("UMR QP created. Returned with cap.max_inline_data = %d", umr_init_attr_ex.cap.max_inline_data);

    ctx->umr_qpx = ibv_qp_to_qp_ex(ctx->umr_qp);
    if (ctx->umr_qpx == NULL) {
        xccl_mhba_error("AMR QP (qpx) creation failed");
        goto umr_qpx_creation_failed;
    }

    // Turning on the IBV_SEND_SIGNALED option, will cause the reported work comletion to be with MLX5DV_WC_UMR opcode.
    // The option IBV_SEND_INLINE is required by the current API.
    ctx->umr_qpx->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

    ctx->umr_mlx5dv_qp_ex = mlx5dv_qp_ex_from_ibv_qp_ex(ctx->umr_qpx);
    if (ctx->umr_mlx5dv_qp_ex == NULL) {
        xccl_mhba_debug("UMR QP (mlx5dv_qp) creation failed");
        goto umr_mlx5dv_qp_creation_failed;
    }
    return XCCL_OK;

    umr_mlx5dv_qp_creation_failed:
    umr_qpx_creation_failed:
    ibv_destroy_qp(ctx->umr_qp);
    umr_qp_creation_failed:
    ibv_destroy_cq(ctx->umr_cq);
    umr_cq_creation_failed:
    return XCCL_ERR_NO_MESSAGE;
}

xccl_status_t init_umr(xccl_mhba_context_t *ctx) {
    struct ibv_port_attr port_attr;
    xccl_status_t status;
    status = create_umr_qp(ctx);
    if (status != XCCL_OK) {
        return status;
    }
    if (ibv_query_port(ctx->umr_qp->context, ctx->ib_port, &port_attr) != 0) {
        xccl_mhba_error("Couldn't get port info");
        return XCCL_ERR_NO_MESSAGE;
    }
    status = remote_qp_connect(ctx->umr_qp, ctx->umr_qp->qp_num, port_attr.lid, ctx->ib_port);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

xccl_status_t
create_matser_key(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node, struct mlx5dv_mkey **mkey_ptr) {
    struct mlx5dv_mkey* mkey;
    xccl_mhba_debug("Create MasterMKey");
    struct mlx5dv_mkey_init_attr umr_mkey_init_attr;
    memset(&umr_mkey_init_attr, 0, sizeof(umr_mkey_init_attr));
    umr_mkey_init_attr.pd = ctx->ib_pd;
    umr_mkey_init_attr.create_flags = MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT;
    // Defines how many entries the Mkey will support.
    // In case the MKey is used as "strided-KLM based MKey", the number
    // of entries that is needed is increased by one because one entry is
    // consumed by the "strided header" (see mlx5dv_wr_post manual).
    umr_mkey_init_attr.max_entries = node->sbgp->group_size + 1;
    mkey = mlx5dv_create_mkey(
            &umr_mkey_init_attr);
    if (mkey == NULL) {
        xccl_mhba_debug("MasterMKey creation failed (errno=%d)", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    xccl_mhba_debug("umr_master_key_dv_mkey: lkey=0x%x", mkey->lkey);
    *mkey_ptr = mkey;
    return XCCL_OK;
}

xccl_status_t init_mkeys(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node) {
    xccl_status_t status;
    status = create_matser_key(ctx, node, &node->send_mkey);
    if (status != XCCL_OK) {
        return status;
    }
    status = create_matser_key(ctx, node, &node->recv_mkey);
    if (status != XCCL_OK) {
        return status;
    }
    xccl_mhba_debug("Creating %d KLM entries for the send mkey", node->sbgp->group_size);
    node->send_mkey_entries = (struct mlx5dv_mr_interleaved *) calloc(node->sbgp->group_size,
                                                                      sizeof(struct mlx5dv_mr_interleaved));
    xccl_mhba_debug("Creating %d Strided-KLM entries for the recv mkey", node->sbgp->group_size);
    node->recv_mkey_entries = (struct mlx5dv_mr_interleaved *) calloc(node->sbgp->group_size,
                                                                      sizeof(struct mlx5dv_mr_interleaved));
    return XCCL_OK;
}

xccl_status_t poll_umr_cq(xccl_mhba_context_t *ctx) {

    int expected_completions_num = 1;
    struct ibv_wc work_completion[1];
    memset(&work_completion, 0, expected_completions_num * sizeof(struct ibv_wc));
    int num_completions;
    int return_code = poll_cq(ctx->umr_cq, expected_completions_num, &num_completions, work_completion);
    if (return_code <= 0) {
        xccl_mhba_error("UMR WQE on UMR QP failed (error %d)", return_code);
        return XCCL_ERR_NO_MESSAGE;
    }

    if (num_completions != expected_completions_num) {
        xccl_mhba_error("UMR CQ didn't return the expected number of completions (expected %d, returned %d)",
                        expected_completions_num, num_completions);
        return XCCL_ERR_NO_MESSAGE;
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
xccl_status_t
populate_mkey(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node, int mem_access_flags, struct mlx5dv_mkey *mkey,
              struct mlx5dv_mr_interleaved *mkey_entries) {
    xccl_status_t status;
    xccl_mhba_debug("Execute the UMR WQE for populating the MasterMKey");
    ibv_wr_start(ctx->umr_qpx);
    ctx->umr_qpx->wr_id = 1; // First (and only) WR todo: check meaning
    int repeat_count = node->sbgp->group_size / node->block_size; // todo check calc
    mlx5dv_wr_mr_interleaved(ctx->umr_mlx5dv_qp_ex, mkey, mem_access_flags,
                             repeat_count, node->sbgp->group_size,
                             mkey_entries);
    status = poll_umr_cq(ctx);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

xccl_status_t
update_mkey_entries(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node, struct mlx5dv_mkey *mkey,
                    struct mlx5dv_mr_interleaved *mkey_entries,
                    void *buffer_start, int mem_access_flags) {
    xccl_status_t status;
    xccl_mhba_debug("Updating Strided-KLM entries for the recv MasterMKey");
    int i;
    for (i = 0; i < node->sbgp->group_size; i++) {
        struct ibv_mr *curr_bf = (struct ibv_mr *) (buffer_start + i * node->size_of_data_unit);
        mkey_entries[i].addr = (uintptr_t) curr_bf->addr; // TODO: Check why conversion to uintptr_t is needed and if it's correct
        mkey_entries[i].bytes_count = node->block_size;
        mkey_entries[i].bytes_skip = 0;
        mkey_entries[i].lkey = curr_bf->lkey;// TODO: For "Recv Side" MasterMKey, shouldn't the entry be populated with the 'rkey' of the DR's memory region?
        xccl_mhba_debug(
                "recv MasterMKey Strided KLM entries[%d]: addr = 0x%x, bytes_count = %d, bytes_skip = %d, lkey=0x%x",
                i, mkey_entries[i].addr, mkey_entries[i].bytes_count,
                mkey_entries[i].bytes_skip, mkey_entries[i].lkey);
    }
    status = populate_mkey(ctx, node, mem_access_flags, mkey, mkey_entries);
    if (status != XCCL_OK) {
        return status;
    }
    return XCCL_OK;
}

xccl_status_t update_mkeys(xccl_mhba_context_t *ctx, xccl_mhba_node_t *node) {
    xccl_status_t status;
    int send_mem_access_flags = 0;
    status = update_mkey_entries(ctx, node, node->send_mkey, node->send_mkey_entries, node->send_umr_data,
                                 send_mem_access_flags);
    if (status != XCCL_OK) {
        xccl_mhba_error("Failed to update send UMR");
        return status;
    }
    int recv_mem_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    status = update_mkey_entries(ctx, node, node->recv_mkey, node->recv_mkey_entries, node->recv_umr_data,
                                 recv_mem_access_flags);
    if (status != XCCL_OK) {
        xccl_mhba_error("Failed to update recv UMR");
        return status;
    }
    return XCCL_OK;
}

xccl_status_t destroy_umr(xccl_mhba_context_t *ctx) {
    ibv_destroy_qp(ctx->umr_qp);
    ibv_destroy_cq(ctx->umr_cq);
    return XCCL_OK;
}

xccl_status_t destroy_mkeys(xccl_mhba_node_t *node) {
    mlx5dv_destroy_mkey(node->send_mkey);
    mlx5dv_destroy_mkey(node->recv_mkey);
    free(node->send_mkey_entries);
    free(node->recv_mkey_entries);
}


