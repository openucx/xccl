/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_ib.h"

xccl_status_t xccl_mhba_create_ibv_ctx(char *ib_devname, struct ibv_context** ctx)
{
    struct ibv_device **dev_list = ibv_get_device_list(NULL);
    struct mlx5dv_context_attr attr = {};
    struct ibv_device *ib_dev;
    if (!ib_devname) {
        /* If no device was specified by name, use by default the first
           available device. */
        ib_dev = *dev_list;
        if (!ib_dev) {
            xccl_mhba_error("No IB devices found");
            return XCCL_ERR_NO_MESSAGE;
        }
    } else {
        int i;
        for (i = 0; dev_list[i]; ++i)
            if (!strcmp(ibv_get_device_name(dev_list[i]), ib_devname))
                break;
        ib_dev = dev_list[i];
        if (!ib_dev) {
            xccl_mhba_error("IB device %s not found", ib_devname);
            return XCCL_ERR_NO_MESSAGE;
        }
    }

    /* Need to open the device with `MLX5DV_CONTEXT_FLAGS_DEVX` flag, as it is
       needed for mlx5dv_create_mkey() (See man pages of mlx5dv_create_mkey()). */

    attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
    *ctx = mlx5dv_open_device(ib_dev, &attr);
    return XCCL_OK;
}

int xccl_mhba_check_port_active(struct ibv_context* ctx, int port_num)
{
    struct ibv_port_attr  port_attr;
    ibv_query_port(ctx, port_num, &port_attr);
    if (port_attr.state == IBV_PORT_ACTIVE &&
        port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        return 1;
    }
    return 0;
}

int xccl_mhba_get_active_port(struct ibv_context* ctx) {
    struct ibv_device_attr device_attr;
    int i;
    ibv_query_device(ctx, &device_attr);
    for (i=1; i<=device_attr.phys_port_cnt; i++) {
        if (xccl_mhba_check_port_active(ctx, i)) {
            return i;
        }
    }
    return -1;
}

xccl_status_t xccl_mhba_qp_connect(struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, int port)
{
    int ret;
    struct ibv_qp_attr qp_attr;

    xccl_mhba_debug("modify QP to INIT");

    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                              IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC |
                              IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE |
                                    IBV_QP_PKEY_INDEX |
                                    IBV_QP_PORT |
                                    IBV_QP_ACCESS_FLAGS) != 0) {
        xccl_mhba_error("QP RESET->INIT failed");
        return XCCL_ERR_NO_MESSAGE;
    }

    xccl_mhba_debug("modify QP to RTR");

    memset((void *)&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = IBV_MTU_4096;
    qp_attr.dest_qp_num = qp_num;
    qp_attr.rq_psn = 0x123;
    qp_attr.min_rnr_timer = 20;
    qp_attr.max_dest_rd_atomic = 1;
    qp_attr.ah_attr.dlid = lid;
    qp_attr.ah_attr.sl = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num = port;
    ret = ibv_modify_qp(qp, &qp_attr, 0
                        | IBV_QP_STATE
                        | IBV_QP_AV
                        | IBV_QP_PATH_MTU
                        | IBV_QP_DEST_QPN
                        | IBV_QP_RQ_PSN
                        | IBV_QP_MAX_DEST_RD_ATOMIC
                        | IBV_QP_MIN_RNR_TIMER);
    if (ret != 0) {
        xccl_mhba_error("QP INIT->RTR failed (error %d)", ret);
        return XCCL_ERR_NO_MESSAGE;
    }

    // Modify QP to RTS
    xccl_mhba_debug("modify QP to RTS");
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 10;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = 0x123;
    qp_attr.max_rd_atomic = 1;
    ret = ibv_modify_qp(qp, &qp_attr, 0
                        | IBV_QP_STATE
                        | IBV_QP_TIMEOUT
                        | IBV_QP_RETRY_CNT
                        | IBV_QP_RNR_RETRY
                        | IBV_QP_SQ_PSN
                        | IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret != 0) {
        xccl_mhba_error("QP RTR->RTS failed");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}
