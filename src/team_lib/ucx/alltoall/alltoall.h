#ifndef ALLTOALL_H_
#define ALLTOALL_H_
#include "../xccl_ucx_lib.h"

xccl_status_t xccl_ucx_alltoall_pairwise_start(xccl_ucx_collreq_t *req);

xccl_status_t xccl_ucx_alltoall_linear_shift_start(xccl_ucx_collreq_t *req);

#endif
