/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"
#include "xccl_mhba_collective.h"
#include "mem_component.h"
#include <ucs/memory/memory_type.h>
#include "core/xccl_team.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>
#include <infiniband/mlx5dv.h>

static ucs_config_field_t xccl_team_lib_mhba_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_team_lib_mhba_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t xccl_tl_mhba_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_mhba_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {"NET_DEVICES", "",
     "Specifies which network device(s) to use",
     ucs_offsetof(xccl_tl_mhba_context_config_t, devices),
     UCS_CONFIG_TYPE_STRING_ARRAY
    },

    {"ASR_CQ_SIZE", "4",
     "Size of the ASR completion queue",
     ucs_offsetof(xccl_tl_mhba_context_config_t, asr_cq_size),
     UCS_CONFIG_TYPE_UINT
    },

    {"ASR_TX_SIZE", "4",
     "Size of the ASR Send queue for RC QP",
     ucs_offsetof(xccl_tl_mhba_context_config_t, asr_tx_size),
     UCS_CONFIG_TYPE_UINT
    },

    {"ASR_RX_SIZE", "4",
     "Size of the ASR Recv queue for RC QP",
     ucs_offsetof(xccl_tl_mhba_context_config_t, asr_rx_size),
     UCS_CONFIG_TYPE_UINT
    },

    {"IB_GLOBAL", "0",
     "Use global ib routing",
     ucs_offsetof(xccl_tl_mhba_context_config_t, ib_global),
     UCS_CONFIG_TYPE_UINT
    },
    {NULL}
};

static xccl_status_t xccl_mhba_lib_open(xccl_team_lib_h self,
                                        xccl_team_lib_config_t *config)
{
    xccl_team_lib_mhba_t        *tl  = ucs_derived_of(self, xccl_team_lib_mhba_t);
    xccl_team_lib_mhba_config_t *cfg = ucs_derived_of(config, xccl_team_lib_mhba_config_t);

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_MHBA");
    xccl_mhba_debug("Team MHBA opened");
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_create_ibv_ctx(char *ib_devname, struct ibv_context** ctx)
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

static int xccl_mhba_check_port_active(struct ibv_context* ctx, int port_num)
{
    struct ibv_port_attr  port_attr;
    ibv_query_port(ctx, port_num, &port_attr);
    if (port_attr.state == IBV_PORT_ACTIVE &&
        port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        return 1;
    }
    return 0;
}
static int xccl_mhba_get_active_port(struct ibv_context* ctx) {
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

static xccl_status_t
xccl_mhba_context_create(xccl_team_lib_h lib, xccl_context_params_t *params,
                         xccl_tl_context_config_t *config,
                         xccl_tl_context_t **context)
{
    xccl_tl_mhba_context_config_t *cfg =
        ucs_derived_of(config, xccl_tl_mhba_context_config_t);
        xccl_mhba_context_t *ctx = malloc(sizeof(*ctx));
    char *ib_devname = NULL;
    char tmp[128];
    int port = -1;
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    if (cfg->devices.count > 0) {
        ib_devname = cfg->devices.names[0];
        char *pos = strstr(ib_devname, ":");
        strncpy(tmp, ib_devname, (int)(pos - ib_devname));
        ib_devname = tmp;
        port = atoi(pos+1);
    }
    if (XCCL_OK != xccl_mhba_create_ibv_ctx(ib_devname, &ctx->ib_ctx)) {
        xccl_mhba_error("failed to allocate ibv_context");
        return XCCL_ERR_NO_MESSAGE;
    }
    if (port == -1) {
        port = xccl_mhba_get_active_port(ctx->ib_ctx);
    }
    ctx->ib_port = port;
    if (-1 == port || !xccl_mhba_check_port_active(ctx->ib_ctx, port)) {
        xccl_mhba_error("no active ports found on %s", ib_devname);
    }
    xccl_mhba_info("using %s:%d", ib_devname, port);

    ctx->ib_pd = ibv_alloc_pd(ctx->ib_ctx);
    if (!ctx->ib_pd) {
        xccl_mhba_error("failed to allocate ib_pd");
        goto pd_alloc_failed;
    }
    memcpy(&ctx->cfg, cfg, sizeof(*cfg));
    *context = &ctx->super;
    return XCCL_OK;
pd_alloc_failed:
    ibv_close_device(ctx->ib_ctx);
}

static xccl_status_t
xccl_mhba_context_destroy(xccl_tl_context_t *context)
{
    xccl_mhba_context_t *team_mhba_ctx =
        ucs_derived_of(context, xccl_mhba_context_t);
    ibv_dealloc_pd(team_mhba_ctx->ib_pd);
    ibv_close_device(team_mhba_ctx->ib_ctx);
    free(team_mhba_ctx);

    return XCCL_OK;
}

static xccl_status_t remote_qp_connect(struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, int port)
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
    qp_attr.ah_attr.port_num = port; // TODO: Why not using port_num argument?
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

static xccl_status_t xccl_mhba_test_net_ctrl(xccl_mhba_team_t *team, xccl_mhba_context_t *ctx)
{
    xccl_status_t status = XCCL_OK;
    int *tmp = malloc(sizeof(int));
    struct ibv_mr *mr = ibv_reg_mr(ctx->ib_pd, tmp, sizeof(int),
                                   IBV_ACCESS_REMOTE_WRITE |
                                   IBV_ACCESS_LOCAL_WRITE);
    *tmp = 0xdeadbeef;

    struct ibv_sge list = {
        .addr	= (uintptr_t)tmp,
        .length = sizeof(int),
        .lkey	= mr->lkey,
    };

    int my_rank = team->net.sbgp->group_rank;
    int peer = (my_rank + 1) % team->net.sbgp->group_size;

    struct ibv_send_wr wr = {
        .wr_id	    = 1,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = IBV_WR_RDMA_WRITE,
        .send_flags = 0,
        .wr.rdma.remote_addr = (uintptr_t)team->net.remote_ctrl[peer].addr +
        sizeof(uint32_t)*my_rank,
        .wr.rdma.rkey = team->net.remote_ctrl[peer].rkey,
    };
    struct ibv_send_wr *bad_wr;
    int ret = ibv_post_send(team->net.qps[peer], &wr, &bad_wr);
    if (ret) {
        xccl_mhba_error("failed to post send during %s", __FUNCTION__);
        status = XCCL_ERR_NO_MESSAGE;
    }
    ibv_dereg_mr(mr);
    free(tmp);

    while (team->net.ctrl[peer] != 0xdeadbeef) { usleep(100) ;}
    xccl_mhba_info("test success");
    return status;
}

static int xccl_sbgp_rank_to_context(int rank, void *rank_mapper_ctx) {
    xccl_sbgp_t *sbgp = (xccl_sbgp_t*)rank_mapper_ctx;
    return xccl_sbgp_rank2ctx(sbgp, rank);
}

static int xccl_sbgp_rank_to_team(int rank, void *rank_mapper_ctx) {
    xccl_sbgp_t *sbgp = (xccl_sbgp_t*)rank_mapper_ctx;
    return xccl_sbgp_rank2team(sbgp, rank);
}

static int
oob_sbgp_allgather(void *sbuf, void *rbuf, size_t len,
                   int myrank, xccl_ep_range_t r, void *coll_context, void **req) {
    xccl_sbgp_t *sbgp = (xccl_sbgp_t*)coll_context;
    xccl_team_t *team = sbgp->team;
    assert(r.type == XCCL_EP_RANGE_UNDEFINED);
    xccl_ep_range_t range = {
        .type      = XCCL_EP_RANGE_CB,
        .ep_num    = sbgp->group_size,
        .cb.cb     = xccl_sbgp_rank_to_team,
        .cb.cb_ctx = (void*)sbgp,
    };
    team->params.oob.allgather(sbuf, rbuf, len, sbgp->group_rank,
                                  range, team->params.oob.coll_context, req);
    return 0;
}

static xccl_status_t
xccl_mhba_team_create_post(xccl_tl_context_t *context,
                           xccl_team_params_t *params,
                           xccl_team_t *base_team,
                           xccl_tl_team_t **team)
{
    xccl_mhba_context_t *ctx = ucs_derived_of(context, xccl_mhba_context_t);
    xccl_mhba_team_t *mhba_team = malloc(sizeof(*mhba_team));
    xccl_sbgp_t *node, *net;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_port_attr port_attr;
    int shmid, i;
    size_t storage_size, local_data_size;
    uint32_t *local_data, *global_data;

    XCCL_TEAM_SUPER_INIT(mhba_team->super, context, params, base_team);
    node = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE);
    mhba_team->node.sbgp = node;
    storage_size = (MHBA_CTRL_SIZE+MHBA_DATA_SIZE) * node->group_size;

    if (0 == node->group_rank) {
        shmid = shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
    }
    xccl_sbgp_oob_bcast(&shmid, sizeof(int), 0, node, params->oob);
    if (shmid == -1) {
        xccl_mhba_error("failed to allocate sysv shm segment for %d bytes",
                        storage_size);
        return XCCL_ERR_NO_RESOURCE;
    }

    mhba_team->node.storage = shmat(shmid, NULL, 0);
    if (0 == node->group_rank) {
        if (shmctl(shmid, IPC_RMID, NULL) == -1) {
            xccl_mhba_error("failed to shmctl IPC_RMID seg %d",
                            shmid);
            return XCCL_ERR_NO_RESOURCE;
        }
    }
    if (mhba_team->node.storage == (void*)(-1)) {
        xccl_mhba_error("failed to shmat seg %d",
                        shmid);
        return XCCL_ERR_NO_RESOURCE;
    }
    mhba_team->node.ctrl = mhba_team->node.storage;
    mhba_team->node.umr_data = (void*)((ptrdiff_t)mhba_team->node.storage +
        node->group_size*MHBA_CTRL_SIZE);
    mhba_team->node.my_ctrl = (void*)((ptrdiff_t)mhba_team->node.ctrl +
        node->group_rank*MHBA_CTRL_SIZE);
    mhba_team->node.my_umr_data = (void*)((ptrdiff_t)mhba_team->node.umr_data +
        node->group_size*MHBA_DATA_SIZE);

    memset(mhba_team->node.my_ctrl, 0, MHBA_CTRL_SIZE);
    xccl_sbgp_oob_barrier(node, params->oob);
    mhba_team->sequence_number = 1;

    net = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE_LEADERS);
    mhba_team->net.sbgp = net;
    mhba_team->net.ctrl = NULL;
    mhba_team->net.ctrl_mr = NULL;
    mhba_team->net.remote_ctrl = NULL;
    if (XCCL_MHBA_IS_ASR(mhba_team)) {
        mhba_team->net.cq = ibv_create_cq(ctx->ib_ctx, ctx->cfg.asr_cq_size, NULL, NULL, 0);
        if (!mhba_team->net.cq) {
            xccl_mhba_error("failed to allocate ASR CQ");
            goto fail;
        }

        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        qp_init_attr.send_cq = mhba_team->net.cq;
        qp_init_attr.recv_cq = mhba_team->net.cq;
        qp_init_attr.cap.max_send_wr = ctx->cfg.asr_tx_size;
        qp_init_attr.cap.max_recv_wr = ctx->cfg.asr_rx_size;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = 0;
        qp_init_attr.qp_type = IBV_QPT_RC;

        mhba_team->net.qps = malloc(sizeof(struct ibv_qp *)*net->group_size);
        if (!mhba_team->net.qps) {
            xccl_mhba_error("failed to allocate asr qps array");
            goto fail;
        }

        local_data_size = (net->group_size+4)*sizeof(uint32_t);
        local_data = malloc(local_data_size);
        if (!local_data) {
            xccl_mhba_error("failed to allocate local data");
            goto local_data_fail;
        }
        global_data = malloc(local_data_size*net->group_size);
        if (!global_data) {
            xccl_mhba_error("failed to allocate global data");
            goto global_data_fail;
        }

        for (i=0; i<net->group_size; i++) {
            mhba_team->net.qps[i] = ibv_create_qp(ctx->ib_pd, &qp_init_attr);
            if (!mhba_team->net.qps[i]) {
                xccl_mhba_error("failed to create qp for dest %d, errno %d",
                                i, errno);
                goto create_qp_failed;
            }
            local_data[i] = mhba_team->net.qps[i]->qp_num;
        }
        mhba_team->net.ctrl = (uint32_t*)calloc(sizeof(uint32_t), net->group_size);
        if (!mhba_team->net.ctrl) {
            xccl_mhba_error("failed to allocate ctrl");
            goto ctrl_fail;
        }

        mhba_team->net.ctrl_mr = ibv_reg_mr(ctx->ib_pd, mhba_team->net.ctrl,
                                            sizeof(uint32_t)*net->group_size,
                                            IBV_ACCESS_REMOTE_WRITE |
                                            IBV_ACCESS_LOCAL_WRITE);
        if (!mhba_team->net.ctrl_mr) {
            xccl_mhba_error("failed to register control data, errno %d", errno);
            goto ctrl_mr_fail;
        }
        ibv_query_port(ctx->ib_ctx, ctx->ib_port, &port_attr);
        local_data[net->group_size] = port_attr.lid;
        local_data[net->group_size+1] = mhba_team->net.ctrl_mr->rkey;
        *((uint64_t*)&local_data[net->group_size+2]) = (uint64_t)(uintptr_t)mhba_team->net.ctrl_mr->addr;

        mhba_team->net.remote_ctrl = calloc(sizeof(*mhba_team->net.remote_ctrl), net->group_size);
        if (!mhba_team->net.remote_ctrl) {
            xccl_mhba_error("failed to allocate remote_ctrl");
            goto remote_ctrl_fail;
        }

        xccl_sbgp_oob_allgather(local_data, global_data, local_data_size, net, params->oob);
        for (i=0; i<net->group_size; i++) {
            uint32_t *remote_data = (uint32_t*)((uintptr_t)global_data + i*local_data_size);
            remote_qp_connect(mhba_team->net.qps[i], remote_data[net->group_rank],
                              remote_data[net->group_size], ctx->ib_port);
            mhba_team->net.remote_ctrl[i].rkey = remote_data[net->group_size+1];
            mhba_team->net.remote_ctrl[i].addr =
                (void*)(uintptr_t)(*((uint64_t*)&remote_data[net->group_size+2]));
        }
        xccl_sbgp_oob_barrier(net, params->oob);
        xccl_mhba_test_net_ctrl(mhba_team, ctx);

        xccl_tl_context_t *ucx_ctx = xccl_get_tl_context(context->ctx, XCCL_TL_UCX);
        if (!ucx_ctx) {
            xccl_mhba_error("failed to find available ucx tl context");
            goto remote_ctrl_fail;
        }

        xccl_oob_collectives_t oob = {
            .allgather    = oob_sbgp_allgather,
            .req_test     = params->oob.req_test,
            .req_free     = params->oob.req_free,
            .coll_context = (void*)mhba_team->net.sbgp,
            .rank         = mhba_team->net.sbgp->group_rank,
            .size         = mhba_team->net.sbgp->group_size,
        };

        xccl_team_params_t team_params = {
            .range.type      = XCCL_EP_RANGE_CB,
            .range.cb.cb     = xccl_sbgp_rank_to_context,
            .range.cb.cb_ctx = (void*)mhba_team->net.sbgp,
            .oob             = oob,
        };

        if (XCCL_OK != ucx_ctx->lib->team_create_post(ucx_ctx, &team_params, base_team,
                                                      &mhba_team->net.ucx_team)) {
            xccl_mhba_error("failed to start ucx team creation");
            goto remote_ctrl_fail;
        }
        while (XCCL_OK != ucx_ctx->lib->team_create_test(mhba_team->net.ucx_team)) {;}

        free(local_data);
        free(global_data);
    }
    *team = &mhba_team->super;
    return XCCL_OK;

remote_ctrl_fail:
    ibv_dereg_mr(mhba_team->net.ctrl_mr);
ctrl_mr_fail:
    free(mhba_team->net.ctrl);
ctrl_fail:
create_qp_failed:
    free(global_data);
global_data_fail:
    free(local_data);
local_data_fail:
    free(mhba_team->net.qps);
fail:
    return XCCL_ERR_NO_MESSAGE;
}

static xccl_status_t
xccl_mhba_team_create_test(xccl_tl_team_t *team)
{
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_team_destroy(xccl_tl_team_t *team)
{
    xccl_mhba_team_t *mhba_team = ucs_derived_of(team, xccl_mhba_team_t);
    int i;
    xccl_mhba_info("destroying team %p", team);
    if (-1 == shmdt(mhba_team->node.storage)) {
        xccl_mhba_error("failed to shmdt %p, errno %d",
                        mhba_team->node.storage, errno);
    }
    if (XCCL_MHBA_IS_ASR(mhba_team)) {
        ibv_dereg_mr(mhba_team->net.ctrl_mr);
        free(mhba_team->net.ctrl);
        free(mhba_team->net.remote_ctrl);
        for (i=0; i<mhba_team->net.sbgp->group_size; i++) {
            ibv_destroy_qp(mhba_team->net.qps[i]);
        }
        free(mhba_team->net.qps);
        ibv_destroy_cq(mhba_team->net.cq);
        mhba_team->net.ucx_team->ctx->lib->team_destroy(mhba_team->net.ucx_team);
    }
    free(team);
    return XCCL_OK;
}

xccl_status_t xccl_mhba_node_fanin(xccl_mhba_team_t *team, int fanin_value, int root)
{
    int i;
    int *ctrl_v;
    if (team->node.sbgp->group_rank != root) {
        ctrl_v = (int*)team->node.my_ctrl;
        int v = *ctrl_v;
        __sync_fetch_and_add(ctrl_v, (v+fanin_value));
    } else {
        for (i=0; i<team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = (int*)((ptrdiff_t)team->node.ctrl + MHBA_CTRL_SIZE*i);
            if (*ctrl_v != fanin_value) {
                return XCCL_INPROGRESS;
            }
        }
    }
    return XCCL_OK;
}

xccl_status_t xccl_mhba_node_fanout(xccl_mhba_team_t *team, int fanout_value, int root)
{
    int i;
    int *ctrl_v;
    if (team->node.sbgp->group_rank != root) {
        ctrl_v = (int*)((ptrdiff_t)team->node.ctrl + MHBA_CTRL_SIZE*root);
        if (*ctrl_v != fanout_value) {
            return XCCL_INPROGRESS;
        }
    } else {
        ctrl_v = (int*)team->node.my_ctrl;
        int v = *ctrl_v;
        __sync_fetch_and_add(ctrl_v, (v+fanout_value));
    }
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_collective_init(xccl_coll_op_args_t *coll_args,
                          xccl_tl_coll_req_t **request,
                          xccl_tl_team_t *team)
{
    xccl_mhba_team_t *mhba_team  = ucs_derived_of(team, xccl_mhba_team_t);
    xccl_mhba_coll_req_t *req;
    xccl_status_t        status;
    ucs_memory_type_t    mem_type;

    status = xccl_mem_component_type(coll_args->buffer_info.src_buffer,
                                     &mem_type);
    if (status != XCCL_OK) {
        xccl_mhba_error("Memtype detection error");
        return XCCL_ERR_INVALID_PARAM;
    }

    if (mem_type == UCS_MEMORY_TYPE_CUDA) {
        return XCCL_ERR_UNSUPPORTED;
    }

    status = xccl_mhba_collective_init_base(coll_args, &req, mhba_team);
    if (status != XCCL_OK) {
        return status;
    }

    switch (coll_args->coll_type) {
    case XCCL_ALLTOALL:
        status = xccl_mhba_alltoall_init(coll_args, req, mhba_team);
        break;
    default:
        status = XCCL_ERR_INVALID_PARAM;
    }

    if (status != XCCL_OK) {
        free(req);
        return status;
    }

    (*request) = &req->super;
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req  = ucs_derived_of(request, xccl_mhba_coll_req_t);
    xccl_schedule_start(&req->schedule);
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req  = ucs_derived_of(request, xccl_mhba_coll_req_t);
    return req->schedule.super.state == XCCL_TASK_STATE_COMPLETED ? XCCL_OK :
        XCCL_INPROGRESS;

}

static xccl_status_t
xccl_mhba_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_mhba_coll_req_t *req = ucs_derived_of(request, xccl_mhba_coll_req_t);
    free(req->tasks);
    free(req);
    return XCCL_OK;
}

xccl_team_lib_mhba_t xccl_team_lib_mhba = {
    .super.name                   = "mhba",
    .super.id                     = XCCL_TL_MHBA,
    .super.priority               = 90,
    .super.team_lib_config        =
    {
        .name                     = "MHBA team library",
        .prefix                   = "TEAM_MHBA_",
        .table                    = xccl_team_lib_mhba_config_table,
        .size                     = sizeof(xccl_team_lib_mhba_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "MHBA tl context",
        .prefix                  = "TEAM_MHBA_",
        .table                   = xccl_tl_mhba_context_config_table,
        .size                    = sizeof(xccl_tl_mhba_context_config_t),
    },
    .super.params.reproducible    = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode     = XCCL_THREAD_MODE_SINGLE | XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage      = XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    .super.params.coll_types      = XCCL_COLL_CAP_ALLTOALL,
    .super.mem_types              = UCS_BIT(UCS_MEMORY_TYPE_HOST),
    .super.ctx_create_mode        = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL,
    .super.team_context_create    = xccl_mhba_context_create,
    .super.team_context_destroy   = xccl_mhba_context_destroy,
    .super.team_context_progress  = NULL,
    .super.team_create_post       = xccl_mhba_team_create_post,
    .super.team_create_test       = xccl_mhba_team_create_test,
    .super.team_destroy           = xccl_mhba_team_destroy,
    .super.team_lib_open          = xccl_mhba_lib_open,
    .super.collective_init        = xccl_mhba_collective_init,
    .super.collective_post        = xccl_mhba_collective_post,
    .super.collective_wait        = NULL,
    .super.collective_test        = xccl_mhba_collective_test,
    .super.collective_finalize    = xccl_mhba_collective_finalize,
    .super.global_mem_map_start   = NULL,
    .super.global_mem_map_test    = NULL,
    .super.global_mem_unmap       = NULL,
};
