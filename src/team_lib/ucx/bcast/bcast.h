#ifndef TEAM_UCX_BCAST_H_
#define TEAM_UCX_BCAST_H_
#include "../xccl_ucx_lib.h"

xccl_status_t xccl_ucx_bcast_linear_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_bcast_knomial_start(xccl_ucx_collreq_t *req);

#endif
