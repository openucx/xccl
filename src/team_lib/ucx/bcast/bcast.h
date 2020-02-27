#ifndef TEAM_UCX_BCAST_H_
#define TEAM_UCX_BCAST_H_
#include "../tccl_ucx_lib.h"

tccl_status_t tccl_ucx_bcast_linear_start(tccl_ucx_collreq_t *req);
tccl_status_t tccl_ucx_bcast_knomial_start(tccl_ucx_collreq_t *req);

#endif
