/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_ib.h"
#include "utils/utils.h"

static pthread_spinlock_t mmio_spinlock;

static __attribute__((constructor)) void lock_constructor(void)
{
    pthread_spin_init(&mmio_spinlock, PTHREAD_PROCESS_PRIVATE);
}

static inline void mmio_wc_spinlock(pthread_spinlock_t *lock)
{
    pthread_spin_lock(lock);
    //#if !defined(__i386__) && !defined(__x86_64__)
    //	/* For x86 the serialization within the spin lock is enough to
    //	 * strongly order WC and other memory types. */
    //	mmio_wc_start();
    //#endif
}

static inline void mmio_wc_spinunlock(pthread_spinlock_t *lock)
{
    /* It is possible that on x86 the atomic in the lock is strong enough
	 * to force-flush the WC buffers quickly, and this SFENCE can be
	 * omitted too. */
    mmio_flush_writes();
    pthread_spin_unlock(lock);
}

static void pthread_mmio_write64_be(void *addr, __be64 val)
{
    __be32 first_dword  = htobe32(be64toh(val) >> 32);
    __be32 second_dword = htobe32(be64toh(val));

    /* The WC spinlock, by definition, provides global ordering for all UC
	   and WC stores within the critical region. */
    mmio_wc_spinlock(&mmio_spinlock);

    mmio_write32_be(addr, first_dword);
    mmio_write32_be(addr + 4, second_dword);

    mmio_wc_spinunlock(&mmio_spinlock);
}

#define HAVE_FUNC_ATTRIBUTE_IFUNC 1

#if HAVE_FUNC_ATTRIBUTE_IFUNC
void mmio_write64_be(void *addr, __be64 val)
        __attribute__((ifunc("resolve_mmio_write64_be")));
static write64_fn_t resolve_mmio_write64_be(void);
#else
__asm__(".type mmio_write64_be, %gnu_indirect_function");
write64_fn_t resolve_mmio_write64_be(void) __asm__("mmio_write64_be");
#endif

write64_fn_t resolve_mmio_write64_be(void)
{
    return &pthread_mmio_write64_be;
}

xccl_status_t
xccl_mhba_ibv_qp_to_mlx5dv_qp(struct ibv_qp *umr_qp, struct internal_qp *mqp)
{
    struct mlx5dv_obj dv_obj = {};
    memset((void *)&dv_obj, 0, sizeof(struct mlx5dv_obj));
    dv_obj.qp.in  = umr_qp;
    dv_obj.qp.out = &mqp->qp;
    mqp->qp_num   = umr_qp->qp_num;
    if (mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_QP)) {
        xccl_mhba_error("mlx5dv_init failed - errno %d", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    mqp->sq_cur_post = 0;
    mqp->sq_qend     = mqp->qp.sq.buf + (mqp->qp.sq.wqe_cnt << SQ_WQE_SHIFT);
    mqp->fm_cache    = 0;
    mqp->sq_start    = mqp->qp.sq.buf;
    mqp->offset      = 0;
    ucs_spinlock_init(&mqp->qp_spinlock, 0);
    return XCCL_OK;
}

xccl_status_t xccl_mhba_destroy_mlxdv_qp(struct internal_qp *mqp)
{
    ucs_spinlock_destroy(&mqp->qp_spinlock);
    return XCCL_OK;
}

static inline void post_send_db(struct internal_qp *mqp, int nreq, void *ctrl)
{
    if (ucs_unlikely(!nreq))
        return;

    /*
	 * Make sure that descriptors are written before
	 * updating doorbell record and ringing the doorbell
	 */
    udma_to_device_barrier();
    mqp->qp.dbrec[MLX5_SND_DBR] = htobe32(mqp->sq_cur_post & 0xffff);

    /* Make sure that the doorbell write happens before the memcpy
	 * to WC memory below
	 */
    mmio_wc_start();

    mmio_write64_be(mqp->qp.bf.reg + mqp->offset, *(__be64 *)ctrl);

    /*
	 * use mmio_flush_writes() to ensure write combining buffers are
	 * flushed out of the running CPU. This must be carried inside
	 * the spinlock. Otherwise, there is a potential race. In the
	 * race, CPU A writes doorbell 1, which is waiting in the WC
	 * buffer. CPU B writes doorbell 2, and it's write is flushed
	 * earlier. Since the mmio_flush_writes is CPU local, this will
	 * result in the HCA seeing doorbell 2, followed by doorbell 1.
	 * Flush before toggling bf_offset to be latency oriented.
	 */
    mmio_flush_writes();
    mqp->offset ^= mqp->qp.bf.size;
}

void xccl_mhba_wr_start(struct internal_qp *mqp)
{
    ucs_spin_lock(&mqp->qp_spinlock);
    mqp->nreq = 0;
}

void xccl_mhba_wr_complete(struct internal_qp *mqp)
{
    post_send_db(mqp, mqp->nreq, mqp->cur_ctrl);
    ucs_spin_unlock(&mqp->qp_spinlock);
}

static inline void
common_wqe_init(struct ibv_qp_ex *ibqp, struct internal_qp *mqp)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    uint8_t fence;
    uint32_t idx;

    idx = mqp->sq_cur_post & (mqp->qp.sq.wqe_cnt - 1);

    ctrl = mqp->sq_start + (idx << MLX5_SEND_WQE_SHIFT);
    *(uint32_t *)((void *)ctrl + 8) = 0;

    fence         = (ibqp->wr_flags & IBV_SEND_FENCE) ? MLX5_WQE_CTRL_FENCE :
                                                        mqp->fm_cache;
    mqp->fm_cache = 0;

    // if any Fence issue - this section has been changed
    ctrl->fm_ce_se =
            fence |
            (ibqp->wr_flags & IBV_SEND_SIGNALED ? MLX5_WQE_CTRL_CQ_UPDATE : 0) |
            (ibqp->wr_flags & IBV_SEND_SOLICITED ? MLX5_WQE_CTRL_SOLICITED : 0);

    ctrl->opmod_idx_opcode = htobe32(((mqp->sq_cur_post & 0xffff) << 8) |
                                     MLX5_OPCODE_UMR);

    mqp->cur_ctrl = ctrl;
}

static inline void common_wqe_finilize(struct internal_qp *mqp)
{
    mqp->cur_ctrl->qpn_ds = htobe32(mqp->cur_size | (mqp->qp_num << 8));

    mqp->sq_cur_post += xccl_round_up(mqp->cur_size, 4);
}

/* The strided block format is as the following:
 * | repeat_block | entry_block | entry_block |...| entry_block |
 * While the repeat entry contains details on the list of the block_entries.
 */
static void
umr_strided_seg_create_noninline(struct internal_qp *mqp, uint32_t repeat_count,
                                 uint16_t num_interleaved,
                                 struct mlx5dv_mr_interleaved *data, void *seg,
                                 void *qend, uint32_t ptr_mkey,
                                 void *ptr_address, int *wqe_size,
                                 int *xlat_size, uint64_t *reglen)
{
    struct mlx5_wqe_umr_pointer_seg *pseg;
    struct mlx5_wqe_umr_repeat_block_seg *rb;
    struct mlx5_wqe_umr_repeat_ent_seg *eb;
    uint64_t byte_count = 0;
    int i;

    /* set pointer segment */
    pseg          = seg;
    pseg->mkey    = htobe32(ptr_mkey);
    pseg->address = htobe64((uint64_t)ptr_address);

    /* set actual repeated and entry blocks segments */
    rb               = ptr_address;
    rb->op           = htobe32(0x400); // PRM header entry - repeated blocks
    rb->reserved     = 0;
    rb->num_ent      = htobe16(num_interleaved);
    rb->repeat_count = htobe32(repeat_count);
    eb               = rb->entries;

    /*
	 * ------------------------------------------------------------
	 * | repeat_block | entry_block | entry_block |...| entry_block
	 * ------------------------------------------------------------
	 */
    for (i = 0; i < num_interleaved; i++, eb++) {
        byte_count += data[i].bytes_count;
        eb->va         = htobe64(data[i].addr);
        eb->byte_count = htobe16(data[i].bytes_count);
        eb->stride     = htobe16(data[i].bytes_count + data[i].bytes_skip);
        eb->memkey     = htobe32(data[i].lkey);
    }

    rb->byte_count = htobe32(byte_count);
    *reglen        = byte_count * repeat_count;
    *wqe_size      = sizeof(struct mlx5_wqe_umr_pointer_seg);
    *xlat_size     = (num_interleaved + 1) * sizeof(*eb);
}

static inline bool check_comp_mask(uint64_t input, uint64_t supported)
{
    return (input & ~supported) == 0;
}

static inline uint8_t get_umr_mr_flags(uint32_t acc)
{
    return ((acc & IBV_ACCESS_REMOTE_ATOMIC ?
                     MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_ATOMIC :
                     0) |
            (acc & IBV_ACCESS_REMOTE_WRITE ?
                     MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_REMOTE_WRITE :
                     0) |
            (acc & IBV_ACCESS_REMOTE_READ ?
                     MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_REMOTE_READ :
                     0) |
            (acc & IBV_ACCESS_LOCAL_WRITE ?
                     MLX5_WQE_MKEY_CONTEXT_ACCESS_FLAGS_LOCAL_WRITE :
                     0));
}

/* External API to expose the non-inline UMR registration */
xccl_status_t xccl_mhba_send_wr_mr_noninline(
        struct internal_qp *mqp, struct mlx5dv_mkey *dv_mkey,
        uint32_t access_flags, uint32_t repeat_count, uint16_t num_entries,
        struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey,
        void *ptr_address, struct ibv_qp_ex *ibqp)
{
    struct mlx5_wqe_umr_ctrl_seg *umr_ctrl_seg;
    struct mlx5_wqe_mkey_context_seg *mk_seg;
    int xlat_size;
    int size;
    uint64_t reglen = 0;
    void *qend      = mqp->sq_qend;
    void *seg;

    if (ucs_unlikely(!check_comp_mask(access_flags,
                                      IBV_ACCESS_LOCAL_WRITE |
                                              IBV_ACCESS_REMOTE_WRITE |
                                              IBV_ACCESS_REMOTE_READ |
                                              IBV_ACCESS_REMOTE_ATOMIC))) {
        xccl_mhba_error("Un-supported UMR flags");
        return XCCL_ERR_NO_MESSAGE;
    }

    common_wqe_init(ibqp, mqp);
    mqp->cur_size      = sizeof(struct mlx5_wqe_ctrl_seg) / 16;
    mqp->cur_ctrl->imm = htobe32(dv_mkey->lkey);
    seg = umr_ctrl_seg = (void *)mqp->cur_ctrl +
                         sizeof(struct mlx5_wqe_ctrl_seg);

    memset(umr_ctrl_seg, 0, sizeof(*umr_ctrl_seg));
    umr_ctrl_seg->mkey_mask = htobe64(
            MLX5_WQE_UMR_CTRL_MKEY_MASK_LEN |
            MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_LOCAL_WRITE |
            MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_READ |
            MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_REMOTE_WRITE |
            MLX5_WQE_UMR_CTRL_MKEY_MASK_ACCESS_ATOMIC |
            MLX5_WQE_UMR_CTRL_MKEY_MASK_FREE);

    seg += sizeof(struct mlx5_wqe_umr_ctrl_seg);
    mqp->cur_size += sizeof(struct mlx5_wqe_umr_ctrl_seg) / 16;

    if (ucs_unlikely(seg == qend))
        seg = mqp->sq_start;

    mk_seg = seg;
    memset(mk_seg, 0, sizeof(*mk_seg));
    mk_seg->access_flags = get_umr_mr_flags(access_flags);
    mk_seg->qpn_mkey     = htobe32(0xffffff00 | (dv_mkey->lkey & 0xff));

    seg += sizeof(struct mlx5_wqe_mkey_context_seg);
    mqp->cur_size += (sizeof(struct mlx5_wqe_mkey_context_seg) / 16);

    if (ucs_unlikely(seg == qend))
        seg = mqp->sq_start;

    umr_strided_seg_create_noninline(mqp, repeat_count, num_entries, data, seg,
                                     qend, ptr_mkey, ptr_address, &size,
                                     &xlat_size, &reglen);

    mk_seg->len                 = htobe64(reglen);
    umr_ctrl_seg->klm_octowords = htobe16(align(xlat_size, 64) / 16);
    mqp->cur_size += size / 16;

    mqp->fm_cache = MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE;
    mqp->nreq++;

    common_wqe_finilize(mqp);
    return XCCL_OK;
}

xccl_status_t xccl_mhba_create_ibv_ctx(char *ib_devname,
                                       struct ibv_context **ctx)
{
    struct ibv_device        **dev_list = ibv_get_device_list(NULL);
    struct mlx5dv_context_attr attr     = {};
    struct ibv_device         *ib_dev;
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
    *ctx       = mlx5dv_open_device(ib_dev, &attr);
    return XCCL_OK;
}

int xccl_mhba_check_port_active(struct ibv_context *ctx, int port_num)
{
    struct ibv_port_attr port_attr;
    ibv_query_port(ctx, port_num, &port_attr);
    if (port_attr.state == IBV_PORT_ACTIVE &&
        port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        return 1;
    }
    return 0;
}

int xccl_mhba_get_active_port(struct ibv_context *ctx)
{
    struct ibv_device_attr device_attr;
    int                    i;
    ibv_query_device(ctx, &device_attr);
    for (i = 1; i <= device_attr.phys_port_cnt; i++) {
        if (xccl_mhba_check_port_active(ctx, i)) {
            return i;
        }
    }
    return -1;
}

xccl_status_t xccl_mhba_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                   uint16_t lid, int port)
{
    int                ret;
    struct ibv_qp_attr qp_attr;

    xccl_mhba_debug("modify QP to INIT");
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state        = IBV_QPS_INIT;
    qp_attr.pkey_index      = 0;
    qp_attr.port_num        = port;
    qp_attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                              IBV_ACCESS_REMOTE_ATOMIC | IBV_ACCESS_LOCAL_WRITE;
    if (ibv_modify_qp(qp, &qp_attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                          IBV_QP_ACCESS_FLAGS) != 0) {
        xccl_mhba_error("QP RESET->INIT failed");
        return XCCL_ERR_NO_MESSAGE;
    }

    xccl_mhba_debug("modify QP to RTR");

    memset((void *)&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state              = IBV_QPS_RTR;
    qp_attr.path_mtu              = IBV_MTU_4096;
    qp_attr.dest_qp_num           = qp_num;
    qp_attr.rq_psn                = 0x123;
    qp_attr.min_rnr_timer         = 20;
    qp_attr.max_dest_rd_atomic    = 1;
    qp_attr.ah_attr.dlid          = lid;
    qp_attr.ah_attr.sl            = 0;
    qp_attr.ah_attr.src_path_bits = 0;
    qp_attr.ah_attr.port_num      = port;
    
    ret = ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE | IBV_QP_AV |
                        IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret != 0) {
        xccl_mhba_error("QP INIT->RTR failed (error %d)", ret);
        return XCCL_ERR_NO_MESSAGE;
    }

    // Modify QP to RTS
    xccl_mhba_debug("modify QP to RTS");
    qp_attr.qp_state      = IBV_QPS_RTS;
    qp_attr.timeout       = 10;
    qp_attr.retry_cnt     = 7;
    qp_attr.rnr_retry     = 7;
    qp_attr.sq_psn        = 0x123;
    qp_attr.max_rd_atomic = 1;
    
    ret = ibv_modify_qp(qp, &qp_attr,  IBV_QP_STATE | IBV_QP_TIMEOUT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret != 0) {
        xccl_mhba_error("QP RTR->RTS failed");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

xccl_status_t xccl_mhba_init_dc_qps_and_connect(xccl_mhba_team_t *mhba_team, uint32_t *local_data, uint8_t port_num){
    int i;
    struct ibv_qp_init_attr_ex attr_ex;
    struct mlx5dv_qp_init_attr attr_dv;
    struct ibv_qp_attr qp_attr_to_init;
    struct ibv_qp_attr qp_attr_to_rtr;
    struct ibv_qp_attr qp_attr_to_rts;
    memset(&attr_ex, 0, sizeof(attr_ex));
    memset(&attr_dv, 0, sizeof(attr_dv));
    memset(&qp_attr_to_init, 0, sizeof(qp_attr_to_init));
    memset(&qp_attr_to_rtr, 0, sizeof(qp_attr_to_rtr));
    memset(&qp_attr_to_rts, 0, sizeof(qp_attr_to_rts));

    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = mhba_team->net.cq;
    attr_ex.recv_cq = mhba_team->net.cq;
    attr_ex.pd = mhba_team->node.shared_pd;
    attr_ex.cap.max_send_wr = (SQUARED(mhba_team->node.sbgp->group_size / 2) + 1) * MAX_OUTSTANDING_OPS *
                              xccl_round_up(mhba_team->net.net_size, mhba_team->num_dci_qps);
    attr_ex.cap.max_send_sge = 1;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_SEND_OPS_FLAGS | IBV_QP_INIT_ATTR_PD;
    attr_ex.send_ops_flags = IBV_QP_EX_WITH_RDMA_WRITE | IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM |
                             IBV_QP_EX_WITH_ATOMIC_FETCH_AND_ADD;
    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC | MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCI;
    attr_dv.create_flags |= MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;

    qp_attr_to_init.qp_state = IBV_QPS_INIT;
    qp_attr_to_init.pkey_index = 0;
    qp_attr_to_init.port_num = port_num;
    qp_attr_to_init.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                                      IBV_ACCESS_REMOTE_READ |
                                      IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;

    qp_attr_to_rtr.qp_state = IBV_QPS_RTR;
    qp_attr_to_rtr.path_mtu = IBV_MTU_4096;
    qp_attr_to_rtr.min_rnr_timer = 20;
    qp_attr_to_rtr.ah_attr.port_num = port_num;
    qp_attr_to_rtr.ah_attr.is_global = 0;

    qp_attr_to_rts.qp_state = IBV_QPS_RTS;
    qp_attr_to_rts.timeout = 10; //todo - what value?
    qp_attr_to_rts.retry_cnt = 7;
    qp_attr_to_rts.rnr_retry = 7;
    qp_attr_to_rts.sq_psn = 0x123;
    qp_attr_to_rts.max_rd_atomic = 1;

    //create DCIs
    for (i =0; i<mhba_team->num_dci_qps ;i++) {
        mhba_team->net.dcis[i].dci_qp = mlx5dv_create_qp(mhba_team->node.shared_ctx, &attr_ex, &attr_dv);
        if (!mhba_team->net.dcis[i].dci_qp) {
            xccl_mhba_error("Couldn't create DCI QP");
            goto fail;
        }
        // Turn DCI ibv_qp to ibv_qpex and ibv_mqpex
        mhba_team->net.dcis[i].dc_qpex = ibv_qp_to_qp_ex(mhba_team->net.dcis[i].dci_qp);
        if (!mhba_team->net.dcis[i].dc_qpex) {
            xccl_mhba_error("Failed turn ibv_qp to ibv_qp_ex, error: %d", errno);
            goto fail;
        }
        mhba_team->net.dcis[i].dc_mqpex = mlx5dv_qp_ex_from_ibv_qp_ex(mhba_team->net.dcis[i].dc_qpex);
        if (!mhba_team->net.dcis[i].dc_mqpex) {
            xccl_mhba_error("Failed turn ibv_qp_ex to mlx5dv_qp_ex, error: %d", errno);
            goto fail;
        }

        if (ibv_modify_qp(mhba_team->net.dcis[i].dci_qp, &qp_attr_to_init,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT) != 0) {
            xccl_mhba_error("Failed to modify init qp");
            goto fail;
        }

        if (ibv_modify_qp(mhba_team->net.dcis[i].dci_qp, &qp_attr_to_rtr,
                          IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV) != 0) {
            xccl_mhba_error("Failed to modify qp to rtr");
            goto fail;
        }

        if (ibv_modify_qp(mhba_team->net.dcis[i].dci_qp, &qp_attr_to_rts,
                          IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY
                          | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) != 0) {
            xccl_mhba_error("Failed to modify qp to rts");
            goto fail;
        }
    }

    //create DCT
    memset(&attr_ex, 0, sizeof(struct ibv_qp_init_attr_ex));
    memset(&attr_dv, 0, sizeof(struct mlx5dv_qp_init_attr));

    attr_ex.qp_type = IBV_QPT_DRIVER;
    attr_ex.send_cq = mhba_team->net.cq;
    attr_ex.recv_cq = mhba_team->net.cq;
    attr_ex.comp_mask |= IBV_QP_INIT_ATTR_PD;
    attr_ex.pd = mhba_team->node.shared_pd;
    struct ibv_srq_init_attr srq_attr;
    memset(&srq_attr, 0, sizeof(struct ibv_srq_init_attr));
    srq_attr.attr.max_wr = 1;
    srq_attr.attr.max_sge = 1;
    // SRQ isn't really needed since we don't use SEND and RDMA WRITE with IMM, but needed because it's DCT
    mhba_team->net.srq = ibv_create_srq(mhba_team->node.shared_pd, &srq_attr);
    if (mhba_team->net.srq  == NULL) {
        xccl_mhba_error("Failed to create Shared Receive Queue (SRQ)");
        goto fail;
    }
    attr_ex.srq = mhba_team->net.srq ;

    attr_dv.comp_mask |= MLX5DV_QP_INIT_ATTR_MASK_DC;
    attr_dv.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    attr_dv.dc_init_attr.dct_access_key = DC_KEY;

    mhba_team->net.dct_qp = mlx5dv_create_qp(mhba_team->node.shared_ctx, &attr_ex, &attr_dv);
    if (mhba_team->net.dct_qp == NULL) {
        xccl_mhba_error("Couldn't create DCT QP errno=%d",errno);
        goto srq_fail;
    }

    if (ibv_modify_qp(mhba_team->net.dct_qp, &qp_attr_to_init,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
        xccl_mhba_error("Failed to modify init qp");
        goto dct_fail;
    }

    if (ibv_modify_qp(mhba_team->net.dct_qp, &qp_attr_to_rtr,
                      IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV| IBV_QP_MIN_RNR_TIMER) != 0) {
        xccl_mhba_error("Failed to modify init qp");
        goto dct_fail;
    }

    local_data[0] = mhba_team->net.dct_qp->qp_num;
    return XCCL_OK;

dct_fail:
    if(ibv_destroy_qp(mhba_team->net.dct_qp)) {
        xccl_mhba_error("Couldn't destroy QP");
    }
srq_fail:
    if(ibv_destroy_srq(mhba_team->net.srq)) {
        xccl_mhba_error("Couldn't destroy SRQ");
    }
fail:
    for (i=i-1; i>= 0;i--) {
        if(ibv_destroy_qp(mhba_team->net.dcis[i].dci_qp)) {
            xccl_mhba_error("Couldn't destroy QP");
        }
    }
    return XCCL_ERR_NO_MESSAGE;
}

xccl_status_t xccl_mhba_create_rc_qps(xccl_mhba_team_t *mhba_team, uint32_t *local_data){
    struct ibv_qp_init_attr qp_init_attr;
    int i;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    //todo change in case of non-homogenous ppn
    qp_init_attr.send_cq = mhba_team->net.cq;
    qp_init_attr.recv_cq = mhba_team->net.cq;
    qp_init_attr.cap.max_send_wr =
            (SQUARED(mhba_team->node.sbgp->group_size / 2) + 1) * MAX_OUTSTANDING_OPS; // TODO switch back to fixed tx/rx
    qp_init_attr.cap.max_recv_wr = 0;
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.cap.max_recv_sge    = 0;
    qp_init_attr.cap.max_inline_data = 0;
    qp_init_attr.qp_type             = IBV_QPT_RC;

    mhba_team->net.rc_qps = malloc(sizeof(struct ibv_qp *) * mhba_team->net.net_size);
    if (!mhba_team->net.rc_qps) {
        xccl_mhba_error("failed to allocate asr qps array");
        goto fail_after_malloc;
    }
    for (i = 0; i < mhba_team->net.net_size; i++) {
        mhba_team->net.rc_qps[i] =
                ibv_create_qp(mhba_team->node.shared_pd, &qp_init_attr);
        if (!mhba_team->net.rc_qps[i]) {
            xccl_mhba_error("failed to create qp for dest %d, errno %d", i,
                            errno);
            goto qp_creation_failure;
        }
        local_data[i] = mhba_team->net.rc_qps[i]->qp_num;
    }
    return XCCL_OK;

qp_creation_failure:
    for (i=i-1; i >= 0; i--) {
        if(ibv_destroy_qp(mhba_team->net.rc_qps[i])) {
            xccl_mhba_error("Couldn't destroy QP");
        }
    }
    free(mhba_team->net.rc_qps);
fail_after_malloc:
    return XCCL_ERR_NO_MESSAGE;
}

xccl_status_t xccl_mhba_create_ah(struct ibv_ah **ah_ptr, uint16_t lid, uint8_t port_num,
        xccl_mhba_team_t *mhba_team){
    struct ibv_ah_attr ah_attr;
    memset(&ah_attr, 0, sizeof(struct ibv_ah_attr));

    ah_attr.dlid           = lid;
    ah_attr.port_num       = port_num;
    ah_attr.is_global     = 0;
    ah_attr.grh.hop_limit  = 0;

    *ah_ptr = ibv_create_ah(mhba_team->node.shared_pd, &ah_attr);
    if (!(*ah_ptr)) {
        xccl_mhba_error("Failed to create ah");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}
