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

struct Bcast_data{
    int  shmid;
    int  net_size;
    char sock_path[L_tmpnam];
};

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

    {"TRANSPOSE", "1",
            "Boolean - with transpose or not",
            ucs_offsetof(xccl_tl_mhba_context_config_t, transpose),
            UCS_CONFIG_TYPE_UINT
    },

    {"TRANSPOSE_HW_LIMITATIONS", "1",
            "Boolean - with transpose hw limitations or not",
            ucs_offsetof(xccl_tl_mhba_context_config_t, transpose_hw_limitations),
            UCS_CONFIG_TYPE_UINT
    },

    {"IB_GLOBAL", "0",
     "Use global ib routing",
     ucs_offsetof(xccl_tl_mhba_context_config_t, ib_global),
     UCS_CONFIG_TYPE_UINT
    },

    {"TRANPOSE_BUF_SIZE", "128k",
     "Size of the pre-allocated transpose buffer",
     ucs_offsetof(xccl_tl_mhba_context_config_t, transpose_buf_size),
     UCS_CONFIG_TYPE_MEMUNITS
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
    if (!ctx){
        xccl_mhba_error("context malloc faild");
        return XCCL_ERR_NO_MEMORY;
    }
    char *ib_devname = NULL;
    char tmp[128];
    int port = -1;
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);
    if (cfg->devices.count > 0) {
        ib_devname = cfg->devices.names[0];
        char *pos = strstr(ib_devname, ":");
        int devname_len = (int)(pos - ib_devname);
        strncpy(tmp, ib_devname, devname_len);
        tmp[devname_len] = '\0';
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
    xccl_mhba_debug("using %s:%d", ib_devname, port);

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
    return XCCL_ERR_NO_MESSAGE;
}

static xccl_status_t
xccl_mhba_context_destroy(xccl_tl_context_t *context)
{
    xccl_mhba_context_t *team_mhba_ctx =
        ucs_derived_of(context, xccl_mhba_context_t);
    if(ibv_dealloc_pd(team_mhba_ctx->ib_pd)){
        xccl_mhba_error("Failed to dealloc PD errno %d", errno);
    }
    ibv_close_device(team_mhba_ctx->ib_ctx);
    free(team_mhba_ctx);
    return XCCL_OK;
}

xccl_status_t xccl_mhba_remote_qp_connect(struct ibv_qp *qp, uint32_t qp_num, uint16_t lid, int port)
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

static void calc_block_size(xccl_mhba_team_t* team){
    int i;
    int block_size = team->node.sbgp->group_size;
    int msg_len = MAX_MSG_SIZE;
    for (i=MHBA_NUM_OF_BLOCKS_SIZE_BINS-1;i>=0;i--){
        while ((block_size * block_size) * msg_len > MAX_TRANSPOSE_SIZE){
            block_size -= 1;
        }
        team->blocks_sizes[i] = block_size;
        msg_len >> 1;
    }
}

struct rank_data {
    int team_rank;
    int sbgp_rank;
};

static int compare_rank_data(const void* a, const void* b)
{
    const struct rank_data *d1 = (const struct rank_data *)a;
    const struct rank_data *d2 = (const struct rank_data *)b;
    return d1->team_rank > d2->team_rank ? 1 : -1;
}

static void build_rank_map(xccl_mhba_team_t *mhba_team)
{
    int i;
    struct rank_data *data = malloc(sizeof(*data)*mhba_team->net.net_size);
    struct rank_data my_data = {
        .team_rank = mhba_team->super.params.oob.rank,
        .sbgp_rank = mhba_team->net.sbgp->group_rank};

    xccl_sbgp_oob_allgather(&my_data, data, sizeof(my_data), mhba_team->net.sbgp,
                            mhba_team->super.params.oob);

    mhba_team->net.rank_map = malloc(sizeof(int)*mhba_team->net.net_size);
    qsort(data, mhba_team->net.net_size, sizeof(*data), compare_rank_data);
    for (i=0; i<mhba_team->net.net_size; i++) {
        mhba_team->net.rank_map[data[i].sbgp_rank] = i;
    }
    free(data);
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
    int i;
    mhba_team->node.asr_rank = 0;//todo check in future if always 0
    mhba_team->transpose = ctx->cfg.transpose;
    mhba_team->transpose_hw_limitations = ctx->cfg.transpose_hw_limitations;
    struct Bcast_data bcastData;
    size_t storage_size, local_data_size;
    uint32_t *local_data, *global_data;
    mhba_team->context = ctx;
    memset(mhba_team->occupied_operations_slots,0,MAX_OUTSTANDING_OPS*sizeof(int));
    mhba_team->size = params->oob.size;
    XCCL_TEAM_SUPER_INIT(mhba_team->super, context, params, base_team);

    node = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE);
    if (node->group_size > MAX_STRIDED_ENTRIES){
        xccl_mhba_error("PPN too large");
        goto fail;
    } // todo temp - phase 1
    net = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE_LEADERS);

    if(net->status == XCCL_SBGP_NOT_EXISTS){
        xccl_mhba_error("Problem with net sbgp");
        goto fail;
    }

    mhba_team->node.sbgp = node;
    mhba_team->net.sbgp = net;

    if (XCCL_MHBA_IS_ASR(mhba_team) && node->group_rank != 0) {
        xccl_mhba_error("ASR group rank isn't 0");
        goto fail;
    }

    if(mhba_team->transpose_hw_limitations){
        mhba_team->max_msg_size = MAX_MSG_SIZE;
    } else{
        u_int64_t min = 0;
        u_int64_t max = ~min;
        //todo check calc
        mhba_team->max_msg_size = (max/MAX_OUTSTANDING_OPS)/(mhba_team->node.sbgp->group_size*mhba_team->size);
    }

    storage_size = (MHBA_CTRL_SIZE+ (2*MHBA_DATA_SIZE)) * node->group_size * MAX_OUTSTANDING_OPS +
            MHBA_CTRL_SIZE*MAX_OUTSTANDING_OPS;

    if (mhba_team->node.asr_rank == node->group_rank) {
        bcastData.shmid = shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
        bcastData.net_size = mhba_team->net.sbgp->group_size;
        tmpnam(bcastData.sock_path); //todo make sure security warning mentioned in tempnam API
    }

    xccl_sbgp_oob_bcast(&bcastData, sizeof(struct Bcast_data), mhba_team->node.asr_rank, node, params->oob);

    xccl_status_t status = xccl_mhba_share_ctx_pd(mhba_team->node.asr_rank, &mhba_team->node, mhba_team->context->ib_ctx->cmd_fd,
                                                  mhba_team->context->ib_pd->handle, ctx, params, bcastData.sock_path);
    if(status != XCCL_OK){
        xccl_mhba_error("Failed to create shared ctx & pd");
        goto fail;
    }

    if (bcastData.shmid == -1) {
        xccl_mhba_error("failed to allocate sysv shm segment for %d bytes",
                        storage_size);
        goto fail_after_share_pd;
    }
    mhba_team->net.net_size = bcastData.net_size;
    if (mhba_team->net.net_size < 2){}
    mhba_team->node.storage = shmat(bcastData.shmid, NULL, 0);
    if (mhba_team->node.asr_rank == node->group_rank) {
        if (shmctl(bcastData.shmid, IPC_RMID, NULL) == -1) {
            xccl_mhba_error("failed to shmctl IPC_RMID seg %d",
                            bcastData.shmid);
            goto fail_after_shmat;
        }
    }
    if (mhba_team->node.storage == (void*)(-1)) {
        xccl_mhba_error("failed to shmat seg %d",
                        bcastData.shmid);
        goto fail_after_shmat;
    }
    for(i=0;i<MAX_OUTSTANDING_OPS;i++){
        mhba_team->node.operations[i].ctrl = mhba_team->node.storage + MHBA_CTRL_SIZE*MAX_OUTSTANDING_OPS
                +MHBA_CTRL_SIZE*node->group_size*i;
        mhba_team->node.operations[i].my_ctrl = (void *) ((ptrdiff_t) mhba_team->node.operations[i].ctrl +
                                                          node->group_rank * MHBA_CTRL_SIZE);
        memset(mhba_team->node.operations[i].my_ctrl, 0, MHBA_CTRL_SIZE);
        *((int*)mhba_team->node.operations[i].my_ctrl) = -1; // because sequence number begin from 0
        mhba_team->node.operations[i].send_umr_data = (void*)((ptrdiff_t)mhba_team->node.storage +
                (node->group_size+1)*MHBA_CTRL_SIZE*MAX_OUTSTANDING_OPS
                        +i*MHBA_DATA_SIZE*node->group_size);
        mhba_team->node.operations[i].my_send_umr_data = (void*)((ptrdiff_t)mhba_team->node.operations[i]
                .send_umr_data + node->group_rank*MHBA_DATA_SIZE);
        mhba_team->node.operations[i].recv_umr_data = (void*)((ptrdiff_t)mhba_team->node.operations[i]
                .send_umr_data + MHBA_DATA_SIZE*node->group_size*MAX_OUTSTANDING_OPS);
        mhba_team->node.operations[i].my_recv_umr_data = (void*)((ptrdiff_t)mhba_team->node.operations[i].recv_umr_data
                + node->group_rank*MHBA_DATA_SIZE);
    }

    xccl_sbgp_oob_barrier(node, params->oob);
    mhba_team->sequence_number = 0;

    mhba_team->net.ctrl_mr = NULL;
    mhba_team->net.remote_ctrl = NULL;
    mhba_team->net.rank_map = NULL;
    calc_block_size(mhba_team);
    mhba_team->transpose_buf_mr = NULL;
    mhba_team->transpose_buf = NULL;
    if (mhba_team->node.asr_rank == node->group_rank) {
        if (mhba_team->transpose) {
            mhba_team->transpose_buf = malloc(ctx->cfg.transpose_buf_size);
            if (!mhba_team->transpose_buf) {
                goto fail_after_shmat;
            }
            mhba_team->transpose_buf_mr = ibv_reg_mr(mhba_team->node.shared_pd, mhba_team->transpose_buf,
                                            ctx->cfg.transpose_buf_size,
                                            IBV_ACCESS_REMOTE_WRITE |
                                            IBV_ACCESS_LOCAL_WRITE);
        }
        build_rank_map(mhba_team);
        xccl_status_t status = xccl_mhba_init_umr(ctx, &mhba_team->node);
        if (status!=XCCL_OK){
            xccl_mhba_error("Failed to init UMR");
            goto fail_after_shmat;
        }

        int asr_cq_size = mhba_team->net.sbgp->group_size*MAX_OUTSTANDING_OPS;

        mhba_team->net.cq = ibv_create_cq(mhba_team->node.shared_ctx, asr_cq_size, NULL, NULL, 0);
        if (!mhba_team->net.cq) {
            xccl_mhba_error("failed to allocate ASR CQ");
            goto fail_after_shmat;
        }

        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        qp_init_attr.send_cq = mhba_team->net.cq;
        qp_init_attr.recv_cq = mhba_team->net.cq;
        //todo change in case of non-homogenous ppn
        qp_init_attr.cap.max_send_wr = (squared(mhba_team->node.sbgp->group_size/2)+1)*MAX_OUTSTANDING_OPS;
        qp_init_attr.cap.max_recv_wr = (squared(mhba_team->node.sbgp->group_size/2)+1)*MAX_OUTSTANDING_OPS;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;
        qp_init_attr.cap.max_inline_data = 0;
        qp_init_attr.qp_type = IBV_QPT_RC;

        mhba_team->net.qps = malloc(sizeof(struct ibv_qp *)*net->group_size);
        if (!mhba_team->net.qps) {
            xccl_mhba_error("failed to allocate asr qps array");
            goto fail_after_cq;
        }
        // for each ASR - qp num, in addition to port lid, ctrl segment rkey and address, recieve mkey rkey
        local_data_size = (net->group_size*sizeof(uint32_t))+sizeof(uint32_t)+2*sizeof(uint32_t)+sizeof(void*);
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
            mhba_team->net.qps[i] = ibv_create_qp(mhba_team->node.shared_pd, &qp_init_attr);
            if (!mhba_team->net.qps[i]) {
                xccl_mhba_error("failed to create qp for dest %d, errno %d",
                                i, errno);
                goto ctrl_fail;
            }
            local_data[i] = mhba_team->net.qps[i]->qp_num;
        }

        mhba_team->net.ctrl_mr = ibv_reg_mr(mhba_team->node.shared_pd, mhba_team->node.storage,
                                            MHBA_CTRL_SIZE*MAX_OUTSTANDING_OPS,
                                            IBV_ACCESS_REMOTE_WRITE |
                                            IBV_ACCESS_REMOTE_READ |
                                            IBV_ACCESS_REMOTE_ATOMIC |
                                            IBV_ACCESS_LOCAL_WRITE);
        if (!mhba_team->net.ctrl_mr) {
            xccl_mhba_error("failed to register control data, errno %d", errno);
            goto ctrl_fail;
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

        status = xccl_mhba_init_mkeys(mhba_team);
        if (status!=XCCL_OK){
            xccl_mhba_error("Failed to init mkeys");
            goto remote_ctrl_fail;
        }

        local_data[net->group_size+4] = mhba_team->node.team_recv_mkey->rkey;

        xccl_sbgp_oob_allgather(local_data, global_data, local_data_size, net, params->oob);
        mhba_team->net.rkeys = (uint32_t*) malloc(sizeof(uint32_t)*mhba_team->net.sbgp->group_size);
        for (i=0; i<net->group_size; i++) {
            uint32_t *remote_data = (uint32_t*)((uintptr_t)global_data + i*local_data_size);
            xccl_mhba_remote_qp_connect(mhba_team->net.qps[i], remote_data[net->group_rank],
                              remote_data[net->group_size], ctx->ib_port);
            mhba_team->net.remote_ctrl[i].rkey = remote_data[net->group_size+1];
            mhba_team->net.remote_ctrl[i].addr =
                (void*)(uintptr_t)(*((uint64_t*)&remote_data[net->group_size+2]));
            mhba_team->net.rkeys[i] = remote_data[net->group_size+4];
        }
        xccl_sbgp_oob_barrier(net, params->oob);

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

        mhba_team->dummy_bf_mr = ibv_reg_mr(mhba_team->node.shared_pd, (void*)&mhba_team->dummy_atomic_buff,
                                            sizeof(mhba_team->dummy_atomic_buff), IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mhba_team->dummy_bf_mr) {
            xccl_mhba_error("Failed to register dummy buff (errno=%d)", errno);
            goto remote_ctrl_fail;
        }

        mhba_team->work_completion = (struct ibv_wc*) malloc(sizeof(struct ibv_wc)*mhba_team->net.sbgp->group_size);
        if (!mhba_team->work_completion) {
            xccl_mhba_error("Failed to allocate wc (errno=%d)", errno);
            goto wc_alloc_fail;
        }
        memset(mhba_team->cq_completions,0,sizeof(mhba_team->cq_completions));
    }
    *team = &mhba_team->super;
    return XCCL_OK;

wc_alloc_fail:
    ibv_dereg_mr(mhba_team->dummy_bf_mr);
remote_ctrl_fail:
    ibv_dereg_mr(mhba_team->net.ctrl_mr);
ctrl_fail:
    free(global_data);
global_data_fail:
    free(local_data);
local_data_fail:
    free(mhba_team->net.qps);
fail_after_cq:
    if (ibv_destroy_cq(mhba_team->net.cq)){
        xccl_mhba_error("net cq destroy failed (errno=%d)", errno);
    }
fail_after_shmat:
    if (-1 == shmdt(mhba_team->node.storage)) {
        xccl_mhba_error("failed to shmdt %p, errno %d",
                        mhba_team->node.storage, errno);
    }
fail_after_share_pd:
    status = xccl_mhba_remove_shared_ctx_pd(mhba_team->node.asr_rank, &mhba_team->node);
    if (status != XCCL_OK){
        xccl_mhba_error("failed removing shared ctx & pd");
    }
fail:
    free(mhba_team);
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
    xccl_status_t status = XCCL_OK;
    xccl_mhba_team_t *mhba_team = ucs_derived_of(team, xccl_mhba_team_t);
    int i;
    xccl_mhba_debug("destroying team %p", team);
    if (-1 == shmdt(mhba_team->node.storage)) {
        xccl_mhba_error("failed to shmdt %p, errno %d",
                        mhba_team->node.storage, errno);
    }
    status = xccl_mhba_remove_shared_ctx_pd(mhba_team->node.asr_rank, &mhba_team->node);
    if (status != XCCL_OK){
        xccl_mhba_error("failed removing shared ctx & pd");
    }
    if (mhba_team->node.asr_rank == mhba_team->node.sbgp->group_rank) {

        status = xccl_mhba_destroy_umr(&mhba_team->node);
        if(status!=XCCL_OK){
            xccl_mhba_error("failed to destroy UMR");
        }

        ibv_dereg_mr(mhba_team->net.ctrl_mr);
        free(mhba_team->net.remote_ctrl);
        for (i=0; i<mhba_team->net.sbgp->group_size; i++) {
            ibv_destroy_qp(mhba_team->net.qps[i]);
        }
        free(mhba_team->net.qps);
        if (ibv_destroy_cq(mhba_team->net.cq)){
            xccl_mhba_error("net cq destroy failed (errno=%d)", errno);
        }
        mhba_team->net.ucx_team->ctx->lib->team_destroy(mhba_team->net.ucx_team);

        status = xccl_mhba_destroy_mkeys(&mhba_team->node, 0);
        if (status!=XCCL_OK){
            xccl_mhba_error("failed to destroy Mkeys");
        }
        free(mhba_team->net.rkeys);
        ibv_dereg_mr(mhba_team->dummy_bf_mr);
        free(mhba_team->work_completion);
        free(mhba_team->net.rank_map);
        if (mhba_team->transpose_buf_mr) {
            ibv_dereg_mr(mhba_team->transpose_buf_mr);
            free(mhba_team->transpose_buf);
        }
    }
    free(team);
    return status;
}

xccl_status_t xccl_mhba_node_fanin(xccl_mhba_team_t *team, xccl_mhba_coll_req_t *request)
{
    int i;
    int *ctrl_v;
    int index = seq_index(request->seq_num);
    if(team->occupied_operations_slots[index] && !request->started){
        return XCCL_INPROGRESS;
    } //wait for slot to be open
    team->occupied_operations_slots[index] = 1;
    request->started = 1;
    xccl_mhba_update_mkeys_entries(&team->node, request); // no option for failure status

    if (team->node.sbgp->group_rank != team->node.asr_rank) {
        *team->node.operations[index].my_ctrl = request->seq_num;
    } else {
        for (i=0; i<team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = (int*)((ptrdiff_t)team->node.operations[index].ctrl + MHBA_CTRL_SIZE*i);
            if (*ctrl_v != request->seq_num) {
                return XCCL_INPROGRESS;
            }
        }
    }
    return XCCL_OK;
}

xccl_status_t xccl_mhba_node_fanout(xccl_mhba_team_t *team, xccl_mhba_coll_req_t *request)
{
    int i;
    int *ctrl_v;
    int index = seq_index(request->seq_num);


    /* First phase of fanout: asr signals it completed local ops
       and other ranks wait for asr */
    if (team->node.sbgp->group_rank == team->node.asr_rank) {
        *team->node.operations[index].my_ctrl = request->seq_num;
    } else {
        ctrl_v = (int*)((ptrdiff_t)team->node.operations[index].ctrl + MHBA_CTRL_SIZE*team->node.asr_rank);
        if (*ctrl_v != request->seq_num) {
            return XCCL_INPROGRESS;
        }
    }

    /*Second phase of fanout: wait for remote atomic counters -
      ie wait for the remote data */
    ctrl_v = (int*)((ptrdiff_t)team->node.storage + MHBA_CTRL_SIZE*index);
    assert(*ctrl_v <= team->net.net_size);
    if ( *ctrl_v != team->net.net_size) {
        return XCCL_INPROGRESS;
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
