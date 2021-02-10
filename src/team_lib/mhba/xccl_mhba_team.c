/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "xccl_mhba_lib.h"
#include "core/xccl_team.h"
#include "xccl_mhba_ib.h"
#include <sys/shm.h>
#include <ucm/api/ucm.h>
#include "utils/utils.h"

typedef struct bcast_data {
    int  shmid;
    int  net_size;
    char sock_path[L_tmpnam];
} bcast_data_t;

static int xccl_sbgp_rank_to_context(int rank, void *rank_mapper_ctx)
{
    xccl_sbgp_t *sbgp = (xccl_sbgp_t *)rank_mapper_ctx;
    return xccl_sbgp_rank2ctx(sbgp, rank);
}

static int xccl_sbgp_rank_to_team(int rank, void *rank_mapper_ctx)
{
    xccl_sbgp_t *sbgp = (xccl_sbgp_t *)rank_mapper_ctx;
    return xccl_sbgp_rank2team(sbgp, rank);
}

static int oob_sbgp_allgather(void *sbuf, void *rbuf, size_t len, int myrank,
                              xccl_ep_range_t r, void *coll_context, void **req)
{
    xccl_sbgp_t *sbgp = (xccl_sbgp_t *)coll_context;
    xccl_team_t *team = sbgp->team;
    assert(r.type == XCCL_EP_RANGE_UNDEFINED);
    xccl_ep_range_t range = {
        .type      = XCCL_EP_RANGE_CB,
        .ep_num    = sbgp->group_size,
        .cb.cb     = xccl_sbgp_rank_to_team,
        .cb.cb_ctx = (void *)sbgp,
    };
    team->params.oob.allgather(sbuf, rbuf, len, sbgp->group_rank, range,
                               team->params.oob.coll_context, req);
    return 0;
}

static void calc_block_size(xccl_mhba_team_t *team)
{
    int i;
    int block_size = team->node.sbgp->group_size;
    int msg_len    = 1;
    for (i = 0; i < MHBA_NUM_OF_BLOCKS_SIZE_BINS; i++) {
        while ((block_size * block_size) * msg_len > MAX_TRANSPOSE_SIZE) {
            block_size -= 1;
        }
        team->blocks_sizes[i] = block_size;
        msg_len = msg_len << 1;
    }
}

struct rank_data {
    int team_rank;
    int sbgp_rank;
};

static int compare_rank_data(const void *a, const void *b)
{
    const struct rank_data *d1 = (const struct rank_data *)a;
    const struct rank_data *d2 = (const struct rank_data *)b;
    return d1->team_rank > d2->team_rank ? 1 : -1;
}

static void build_rank_map(xccl_mhba_team_t *mhba_team)
{
    int               i;
    struct rank_data *data    = malloc(sizeof(*data) * mhba_team->net.net_size);
    struct rank_data  my_data = {
        .team_rank = mhba_team->super.params.oob.rank,
        .sbgp_rank = mhba_team->net.sbgp->group_rank
    };
    xccl_sbgp_oob_allgather(&my_data, data, sizeof(my_data),
                            mhba_team->net.sbgp, mhba_team->super.params.oob);

    mhba_team->net.rank_map = malloc(sizeof(int) * mhba_team->net.net_size);
    qsort(data, mhba_team->net.net_size, sizeof(*data), compare_rank_data);
    for (i = 0; i < mhba_team->net.net_size; i++) {
        mhba_team->net.rank_map[data[i].sbgp_rank] = i;
    }
    free(data);
}

static ucs_status_t rcache_reg_mr(void *context, ucs_rcache_t *rcache, void *arg,
                                  ucs_rcache_region_t *rregion, uint16_t flags)
{
    xccl_mhba_team_t *team    = (xccl_mhba_team_t*)context;
    void *addr                = (void*)rregion->super.start;
    size_t length             = (size_t)(rregion->super.end - rregion->super.start);
    xccl_mhba_reg_t* mhba_reg = xccl_rcache_ucs_get_reg_data(rregion);
    int* change_flag          = (int*) arg;

    mhba_reg->region = rregion;
    *change_flag     = 1;
    mhba_reg->mr     = ibv_reg_mr(team->node.shared_pd, addr, length,
                                  (rregion->prot == PROT_WRITE) ? IBV_ACCESS_LOCAL_WRITE
                                  | IBV_ACCESS_REMOTE_WRITE : 0);
    if (!mhba_reg->mr) {
        xccl_mhba_error("Failed to register memory");
        return UCS_ERR_NO_MESSAGE;
    }
    return UCS_OK;
}

static void rcache_dereg_mr(void *context, ucs_rcache_t *rcache,
                            ucs_rcache_region_t *rregion)
{
    xccl_mhba_reg_t* mhba_reg = xccl_rcache_ucs_get_reg_data(rregion);
    assert(mhba_reg->region == rregion);
    ibv_dereg_mr(mhba_reg->mr);
    mhba_reg->mr = NULL;
}

static xccl_status_t create_rcache(xccl_mhba_team_t* mhba_team)
{
    static ucs_rcache_ops_t rcache_ucs_ops = {
            .mem_reg     = rcache_reg_mr,
            .mem_dereg   = rcache_dereg_mr,
            .dump_region = NULL
    };

    ucs_rcache_params_t rcache_params;
    rcache_params.region_struct_size = sizeof(ucs_rcache_region_t)+sizeof(xccl_mhba_reg_t);
    rcache_params.alignment          = UCS_PGT_ADDR_ALIGN;
    rcache_params.max_alignment      = ucs_get_page_size();
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED |
                                       UCM_EVENT_MEM_TYPE_FREE;
    rcache_params.ucm_event_priority = 1000;
    rcache_params.context            = (void*)mhba_team;
    rcache_params.ops                = &rcache_ucs_ops;

    ucs_status_t status = ucs_rcache_create(&rcache_params, "reg cache",
                                            ucs_stats_get_root(), &mhba_team->rcache);

    if (status != UCS_OK) {
        xccl_mhba_error("Failed to create reg cache");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

static inline int xccl_mhba_calc_max_block_size(){
    int block_row_size = 0;
    while((SQUARED(block_row_size + 1) * MAX_MSG_SIZE) <= MAX_TRANSPOSE_SIZE){
        block_row_size += 1;
    }
    return block_row_size;
}

xccl_status_t xccl_mhba_team_create_post(xccl_tl_context_t  *context,
                                         xccl_team_params_t *params,
                                         xccl_team_t        *base_team,
                                         xccl_tl_team_t    **team)
{
    xccl_mhba_context_t    *ctx       = ucs_derived_of(context, xccl_mhba_context_t);
    xccl_mhba_team_t       *mhba_team = malloc(sizeof(*mhba_team));
    xccl_sbgp_t            *node, *net;
    xccl_status_t           status;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_port_attr    port_attr;
    size_t                  storage_size, local_data_size;
    uint32_t               *local_data, *global_data;
    bcast_data_t            bcast_data;
    int                     i, j, net_size, node_size, asr_cq_size;
    mhba_team->node.asr_rank            = 0; //todo check in future if always 0
    mhba_team->transpose                = ctx->cfg.transpose;
    mhba_team->context                  = ctx;
    mhba_team->size                     = params->oob.size;
    mhba_team->sequence_number          = 0;
    mhba_team->net.ctrl_mr              = NULL;
    mhba_team->net.remote_ctrl          = NULL;
    mhba_team->net.rank_map             = NULL;
    mhba_team->transpose_buf_mr         = NULL;
    mhba_team->transpose_buf            = NULL;

    XCCL_TEAM_SUPER_INIT(mhba_team->super, context, params, base_team);

    memset(mhba_team->op_busy, 0, MAX_OUTSTANDING_OPS * sizeof(int));

    if(XCCL_OK != create_rcache(mhba_team)){
        goto fail;
    }

    node = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE);
    if (node->group_size > MAX_STRIDED_ENTRIES) {
        xccl_mhba_error("PPN too large");
        goto fail;
    } // todo temp - phase 1
    node_size = node->group_size;
    net = xccl_team_topo_get_sbgp(base_team->topo, XCCL_SBGP_NODE_LEADERS);

    if (net->status == XCCL_SBGP_NOT_EXISTS) {
        xccl_mhba_error("Problem with net sbgp");
        goto fail;
    }

    mhba_team->node.sbgp = node;
    mhba_team->net.sbgp  = net;

    assert(mhba_team->net.sbgp->status == XCCL_SBGP_ENABLED ||
           node->group_rank != 0);

    mhba_team->max_msg_size = MAX_MSG_SIZE;

    mhba_team->max_num_of_columns = xccl_round_up(node->group_size, xccl_mhba_calc_max_block_size());

    storage_size = (MHBA_CTRL_SIZE + (2 * MHBA_DATA_SIZE * mhba_team->max_num_of_columns)) * node_size *
                       MAX_OUTSTANDING_OPS +
                   MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS;

    if (mhba_team->node.asr_rank == node->group_rank) {
        bcast_data.shmid = shmget(IPC_PRIVATE, storage_size, IPC_CREAT | 0600);
        bcast_data.net_size = mhba_team->net.sbgp->group_size;
        tmpnam(bcast_data.sock_path); //TODO switch to mkstemp
    }

    xccl_sbgp_oob_bcast(&bcast_data, sizeof(bcast_data_t),
                        mhba_team->node.asr_rank, node, params->oob);
    net_size = bcast_data.net_size;
    status   = xccl_mhba_share_ctx_pd(mhba_team, bcast_data.sock_path);
    if (status != XCCL_OK) {
        xccl_mhba_error("Failed to create shared ctx & pd");
        goto fail;
    }

    if (bcast_data.shmid == -1) {
        xccl_mhba_error("failed to allocate sysv shm segment for %d bytes",
                        storage_size);
        goto fail_after_share_pd;
    }
    mhba_team->net.net_size = bcast_data.net_size;
    mhba_team->node.storage = shmat(bcast_data.shmid, NULL, 0);
    if (mhba_team->node.asr_rank == node->group_rank) {
        if (shmctl(bcast_data.shmid, IPC_RMID, NULL) == -1) {
            xccl_mhba_error("failed to shmctl IPC_RMID seg %d",
                            bcast_data.shmid);
            goto fail_after_shmat;
        }
    }
    if (mhba_team->node.storage == (void *)(-1)) {
        xccl_mhba_error("failed to shmat seg %d", bcast_data.shmid);
        goto fail_after_shmat;
    }
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        xccl_mhba_op_t *op = &mhba_team->node.ops[i];

        op->my_recv_umr_data = (void **) malloc(sizeof(void*) * mhba_team->max_num_of_columns);
        op->my_send_umr_data = (void **) malloc(sizeof(void*) * mhba_team->max_num_of_columns);
        op->send_umr_data = (void **) malloc(sizeof(void*) * mhba_team->max_num_of_columns);
        op->recv_umr_data = (void **) malloc(sizeof(void*) * mhba_team->max_num_of_columns);
        if(!op->my_recv_umr_data || !op->my_send_umr_data || !op->send_umr_data || !op->recv_umr_data){
            xccl_mhba_error("malloc failed");
            goto fail_ptr_malloc;
        }

        op->ctrl           = mhba_team->node.storage +
                   MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS +
                   MHBA_CTRL_SIZE * node_size * i;
        op->my_ctrl =
            (xccl_mhba_ctrl_t *)((ptrdiff_t)op->ctrl + node->group_rank * MHBA_CTRL_SIZE);
        op->my_ctrl->mkey_cache_flag = 0;
        if (mhba_team->node.asr_rank == node->group_rank) {
            for (j = 0; j < node->group_size; j++) {
                xccl_mhba_ctrl_t* rank_ctrl = (xccl_mhba_ctrl_t *) ((ptrdiff_t) op->ctrl + j * MHBA_CTRL_SIZE);
                rank_ctrl->seq_num = -1; // because sequence number begin from 0
            }
        }
        for(j=0;j<mhba_team->max_num_of_columns;j++) {
            op->send_umr_data[j] =
                    (void *) ((ptrdiff_t) mhba_team->node.storage +
                              (node_size + 1) * MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS +
                              i * MHBA_DATA_SIZE * mhba_team->max_num_of_columns * node_size + j * MHBA_DATA_SIZE * node_size);
            op->my_send_umr_data[j] = (void *) ((ptrdiff_t) op->send_umr_data[j] +
                                             node->group_rank * MHBA_DATA_SIZE);
            op->recv_umr_data[j] =
                    (void *) ((ptrdiff_t) op->send_umr_data[j] +
                              MHBA_DATA_SIZE * mhba_team->max_num_of_columns * node_size * MAX_OUTSTANDING_OPS);
            op->my_recv_umr_data[j] = (void *) ((ptrdiff_t) op->recv_umr_data[j] +
                                             node->group_rank * MHBA_DATA_SIZE);
        }
    }

    calc_block_size(mhba_team);
    mhba_team->requested_block_size = ctx->cfg.block_size;
    if (mhba_team->node.asr_rank == node->group_rank) {
        for(i=0;i<MAX_OUTSTANDING_OPS;i++) {
            mhba_team->previous_msg_size[i] = 0;
        }
        if (mhba_team->transpose) {
            mhba_team->transpose_buf = malloc(ctx->cfg.transpose_buf_size);
            if (!mhba_team->transpose_buf) {
                goto fail_ptr_malloc;
            }
            mhba_team->transpose_buf_mr =
                ibv_reg_mr(mhba_team->node.shared_pd, mhba_team->transpose_buf,
                           ctx->cfg.transpose_buf_size,
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE);
        }
        build_rank_map(mhba_team);
        status = xccl_mhba_init_umr(ctx, &mhba_team->node);
        if (status != XCCL_OK) {
            xccl_mhba_error("Failed to init UMR");
            goto fail_after_transpose_reg;
        }
        asr_cq_size       = net_size * MAX_OUTSTANDING_OPS;
        mhba_team->net.cq = ibv_create_cq(mhba_team->node.shared_ctx,
                                          asr_cq_size, NULL, NULL, 0);
        if (!mhba_team->net.cq) {
            xccl_mhba_error("failed to allocate ASR CQ");
            goto fail_after_transpose_reg;
        }

        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        //todo change in case of non-homogenous ppn
        qp_init_attr.send_cq = mhba_team->net.cq;
        qp_init_attr.recv_cq = mhba_team->net.cq;
        qp_init_attr.cap.max_send_wr =
            (SQUARED(node_size / 2) + 1) * MAX_OUTSTANDING_OPS; // TODO switch back to fixed tx/rx
        qp_init_attr.cap.max_recv_wr =
            (SQUARED(node_size / 2) + 1) * MAX_OUTSTANDING_OPS;
        qp_init_attr.cap.max_send_sge    = 1;
        qp_init_attr.cap.max_recv_sge    = 1;
        qp_init_attr.cap.max_inline_data = 0;
        qp_init_attr.qp_type             = IBV_QPT_RC;

        mhba_team->net.qps = malloc(sizeof(struct ibv_qp *) * net_size);
        if (!mhba_team->net.qps) {
            xccl_mhba_error("failed to allocate asr qps array");
            goto fail_after_cq;
        }
        // for each ASR - qp num, in addition to port lid, ctrl segment rkey and address, recieve mkey rkey
        local_data_size = (net_size * sizeof(uint32_t)) + sizeof(uint32_t) +
                          2 * sizeof(uint32_t) + sizeof(void *);
        local_data = malloc(local_data_size);
        if (!local_data) {
            xccl_mhba_error("failed to allocate local data");
            goto local_data_fail;
        }
        global_data = malloc(local_data_size * net_size);
        if (!global_data) {
            xccl_mhba_error("failed to allocate global data");
            goto global_data_fail;
        }

        for (i = 0; i < net_size; i++) {
            mhba_team->net.qps[i] =
                ibv_create_qp(mhba_team->node.shared_pd, &qp_init_attr);
            if (!mhba_team->net.qps[i]) {
                xccl_mhba_error("failed to create qp for dest %d, errno %d", i,
                                errno);
                goto ctrl_fail;
            }
            local_data[i] = mhba_team->net.qps[i]->qp_num;
        }

        mhba_team->net.ctrl_mr =
            ibv_reg_mr(mhba_team->node.shared_pd, mhba_team->node.storage,
                       MHBA_CTRL_SIZE * MAX_OUTSTANDING_OPS,
                       IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                       IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE);
        if (!mhba_team->net.ctrl_mr) {
            xccl_mhba_error("failed to register control data, errno %d", errno);
            goto ctrl_fail;
        }
        ibv_query_port(ctx->ib_ctx, ctx->ib_port, &port_attr);
        local_data[net_size]     = port_attr.lid;
        local_data[net_size + 1] = mhba_team->net.ctrl_mr->rkey;
        *((uint64_t *)&local_data[net_size + 2]) =
            (uint64_t)(uintptr_t)mhba_team->net.ctrl_mr->addr;

        mhba_team->net.remote_ctrl =
            calloc(sizeof(*mhba_team->net.remote_ctrl), net_size);
        if (!mhba_team->net.remote_ctrl) {
            xccl_mhba_error("failed to allocate remote_ctrl");
            goto remote_ctrl_fail;
        }

        status = xccl_mhba_init_mkeys(mhba_team);
        if (status != XCCL_OK) {
            xccl_mhba_error("Failed to init mkeys");
            goto remote_ctrl_fail;
        }

        local_data[net_size + 4] = mhba_team->node.team_recv_mkey->rkey;

        xccl_sbgp_oob_allgather(local_data, global_data, local_data_size, net,
                                params->oob);
        mhba_team->net.rkeys = (uint32_t *)malloc(sizeof(uint32_t) * net_size);
        for (i = 0; i < net_size; i++) {
            uint32_t *remote_data =
                (uint32_t *)((uintptr_t)global_data + i * local_data_size);
            xccl_mhba_qp_connect(mhba_team->net.qps[i],
                                 remote_data[net->group_rank],
                                 remote_data[net_size], ctx->ib_port);
            mhba_team->net.remote_ctrl[i].rkey = remote_data[net_size + 1];
            mhba_team->net.remote_ctrl[i].addr =
                (void *)(uintptr_t)(*((uint64_t *)&remote_data[net_size + 2]));
            mhba_team->net.rkeys[i] = remote_data[net_size + 4];
        }

        xccl_tl_context_t *ucx_ctx = xccl_get_tl_context(context->ctx, XCCL_TL_UCX);
        if (!ucx_ctx) {
            xccl_mhba_error("failed to find available ucx tl context");
            goto remote_ctrl_fail;
        }

        xccl_oob_collectives_t oob = {
            .allgather    = oob_sbgp_allgather,
            .req_test     = params->oob.req_test,
            .req_free     = params->oob.req_free,
            .coll_context = (void *)mhba_team->net.sbgp,
            .rank         = mhba_team->net.sbgp->group_rank,
            .size         = mhba_team->net.sbgp->group_size,
        };

        xccl_team_params_t team_params = {
            .range.type      = XCCL_EP_RANGE_CB,
            .range.cb.cb     = xccl_sbgp_rank_to_context,
            .range.cb.cb_ctx = (void *)mhba_team->net.sbgp,
            .oob             = oob,
        };

        if (XCCL_OK !=
            ucx_ctx->lib->team_create_post(ucx_ctx, &team_params, base_team,
                                           &mhba_team->net.ucx_team)) {
            xccl_mhba_error("failed to start ucx team creation");
            goto remote_ctrl_fail;
        }
        while (XCCL_OK !=
               ucx_ctx->lib->team_create_test(mhba_team->net.ucx_team)) {
            ; //TODO make non-blocking
        }

        free(local_data);
        free(global_data);

        mhba_team->dummy_bf_mr = ibv_reg_mr(
            mhba_team->node.shared_pd, (void *)&mhba_team->dummy_atomic_buff,
            sizeof(mhba_team->dummy_atomic_buff),
            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mhba_team->dummy_bf_mr) {
            xccl_mhba_error("Failed to register dummy buff (errno=%d)", errno);
            goto remote_ctrl_fail;
        }

        mhba_team->work_completion =
            (struct ibv_wc *)malloc(sizeof(struct ibv_wc) * net_size);
        if (!mhba_team->work_completion) {
            xccl_mhba_error("Failed to allocate wc (errno=%d)", errno);
            goto wc_alloc_fail;
        }
        memset(mhba_team->cq_completions, 0, sizeof(mhba_team->cq_completions));
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
    if (ibv_destroy_cq(mhba_team->net.cq)) {
        xccl_mhba_error("net cq destroy failed (errno=%d)", errno);
    }
fail_after_transpose_reg:
    ibv_dereg_mr(mhba_team->transpose_buf_mr);
    free(mhba_team->transpose_buf);
fail_ptr_malloc:
    for (i; i >= 0; i--) {
        xccl_mhba_op_t *op = &mhba_team->node.ops[i];
        if (op->recv_umr_data) {
            free(op->recv_umr_data);
        }
        if (op->send_umr_data) {
            free(op->send_umr_data);
        }
        if (op->my_send_umr_data) {
            free(op->my_send_umr_data);
        }
        if (op->my_recv_umr_data) {
            free(op->my_recv_umr_data);
        }
    }
fail_after_shmat:
    if (-1 == shmdt(mhba_team->node.storage)) {
        xccl_mhba_error("failed to shmdt %p, errno %d", mhba_team->node.storage,
                        errno);
    }
fail_after_share_pd:
    status = xccl_mhba_remove_shared_ctx_pd(mhba_team);
    if (status != XCCL_OK) {
        xccl_mhba_error("failed removing shared ctx & pd");
    }
fail:
    free(mhba_team);
    return XCCL_ERR_NO_MESSAGE;
}

xccl_status_t xccl_mhba_team_create_test(xccl_tl_team_t *team)
{
    return XCCL_OK;
}

xccl_status_t xccl_mhba_team_destroy(xccl_tl_team_t *team)
{
    xccl_status_t     status    = XCCL_OK;
    xccl_mhba_team_t *mhba_team = ucs_derived_of(team, xccl_mhba_team_t);
    int               i;
    xccl_mhba_debug("destroying team %p", team);
    ucs_rcache_destroy(mhba_team->rcache);
    if (-1 == shmdt(mhba_team->node.storage)) {
        xccl_mhba_error("failed to shmdt %p, errno %d", mhba_team->node.storage,
                        errno);
    }
    status = xccl_mhba_remove_shared_ctx_pd(mhba_team);
    if (status != XCCL_OK) {
        xccl_mhba_error("failed removing shared ctx & pd");
    }
    for (i = 0; i < MAX_OUTSTANDING_OPS; i++) {
        xccl_mhba_op_t *op = &mhba_team->node.ops[i];
        free(op->recv_umr_data);
        free(op->send_umr_data);
        free(op->my_send_umr_data);
        free(op->my_recv_umr_data);
    }
    if (mhba_team->node.asr_rank == mhba_team->node.sbgp->group_rank) {
        status = xccl_mhba_destroy_umr(&mhba_team->node);
        if (status != XCCL_OK) {
            xccl_mhba_error("failed to destroy UMR");
        }
        ibv_dereg_mr(mhba_team->net.ctrl_mr);
        free(mhba_team->net.remote_ctrl);
        for (i = 0; i < mhba_team->net.sbgp->group_size; i++) {
            ibv_destroy_qp(mhba_team->net.qps[i]);
        }
        free(mhba_team->net.qps);
        if (ibv_destroy_cq(mhba_team->net.cq)) {
            xccl_mhba_error("net cq destroy failed (errno=%d)", errno);
        }
        mhba_team->net.ucx_team->ctx->lib->team_destroy(
            mhba_team->net.ucx_team);

        status = xccl_mhba_destroy_mkeys(mhba_team, 0);
        if (status != XCCL_OK) {
            xccl_mhba_error("failed to destroy Mkeys");
        }
        free(mhba_team->net.rkeys);
        ibv_dereg_mr(mhba_team->dummy_bf_mr);
        free(mhba_team->work_completion);
        free(mhba_team->net.rank_map);
        if (mhba_team->transpose) {
            ibv_dereg_mr(mhba_team->transpose_buf_mr);
            free(mhba_team->transpose_buf);
        }
    }
    free(team);
    return status;
}
