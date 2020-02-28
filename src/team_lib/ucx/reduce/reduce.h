#ifndef REDUCE_H_
#define REDUCE_H_
#include "../xccl_ucx_lib.h"

xccl_status_t xccl_ucx_reduce_linear_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_reduce_linear_progress(xccl_ucx_collreq_t *req);
#endif
