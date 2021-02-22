/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_MHBA_IB_H_
#define XCCL_TEAM_LIB_MHBA_IB_H_
#include "xccl_mhba_lib.h"

#define DC_KEY 1

int xccl_mhba_get_active_port(struct ibv_context *ctx);
int xccl_mhba_check_port_active(struct ibv_context *ctx, int port_num);
xccl_status_t xccl_mhba_create_ibv_ctx(char *ib_devname,
                                       struct ibv_context **ctx);
xccl_status_t xccl_mhba_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                   uint16_t lid, int port);
xccl_status_t xccl_mhba_init_dc_qps_and_connect(xccl_mhba_team_t *mhba_team,
                                                    uint32_t *local_data, uint8_t port_num);
xccl_status_t xccl_mhba_create_rc_qps(xccl_mhba_team_t *mhba_team, uint32_t *local_data);
#endif
