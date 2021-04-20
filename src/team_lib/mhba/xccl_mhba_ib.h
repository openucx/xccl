/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_MHBA_IB_H_
#define XCCL_TEAM_LIB_MHBA_IB_H_
#include "xccl_mhba_lib.h"
#include <stdatomic.h>

#define SQ_WQE_SHIFT 6
#define mmio_flush_writes()                                                    \
    asm volatile("sfence" ::: "memory") // TODO only for __x86_64__
#define udma_to_device_barrier()                                               \
    asm volatile("" ::: "memory") // TODO only for __x86_64__
#define mmio_wc_start() mmio_flush_writes()

#ifdef __CHECKER__
#define __force __attribute__((force))
#else
#define __force
#endif

typedef void (*write64_fn_t)(void *, __be64);

#define MAKE_WRITE(_NAME_, _SZ_)                                               \
    static inline void _NAME_##_be(void *addr, __be##_SZ_ value)               \
    {                                                                          \
        atomic_store_explicit((_Atomic(uint##_SZ_##_t) *)addr,                 \
                              (__force uint##_SZ_##_t)value,                   \
                              memory_order_relaxed);                           \
    }

MAKE_WRITE(mmio_write32, 32)

/* UMR pointer to KLMs/MTTs/RepeatBlock and BSFs location (when inline = 0) */
struct mlx5_wqe_umr_pointer_seg {
    __be32 reserved;
    __be32 mkey;
    __be64 address;
};

struct internal_qp;
typedef struct xccl_mhba_team xccl_mhba_team_t;

xccl_status_t
xccl_mhba_ibv_qp_to_mlx5dv_qp(struct ibv_qp *umr_qp, struct internal_qp *mqp);
void xccl_mhba_wr_start(struct internal_qp *mqp);
xccl_status_t xccl_mhba_send_wr_mr_noninline(
        struct internal_qp *mqp, struct mlx5dv_mkey *dv_mkey,
        uint32_t access_flags, uint32_t repeat_count, uint16_t num_entries,
        struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey,
        void *ptr_address, struct ibv_qp_ex *ibqp);
void xccl_mhba_wr_complete(struct internal_qp *mqp);
int xccl_mhba_get_active_port(struct ibv_context *ctx);
int xccl_mhba_check_port_active(struct ibv_context *ctx, int port_num);
xccl_status_t xccl_mhba_create_ibv_ctx(char *ib_devname,
                                       struct ibv_context **ctx);
xccl_status_t xccl_mhba_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                   uint16_t lid, int port);
xccl_status_t xccl_mhba_init_dc_qps_and_connect(xccl_mhba_team_t *mhba_team,
                                                    uint32_t *local_data, uint8_t port_num);
xccl_status_t xccl_mhba_create_rc_qps(xccl_mhba_team_t *mhba_team, uint32_t *local_data);
xccl_status_t xccl_mhba_create_ah(struct ibv_ah **ah_ptr, uint16_t lid, uint8_t port_num,
                                         xccl_mhba_team_t *mhba_team);
#endif
