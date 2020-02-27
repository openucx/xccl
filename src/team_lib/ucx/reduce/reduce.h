#ifndef REDUCE_H_
#define REDUCE_H_
#include "../tccl_ucx_lib.h"

tccl_status_t tccl_ucx_reduce_linear_start(tccl_ucx_collreq_t *req);
tccl_status_t tccl_ucx_reduce_linear_progress(tccl_ucx_collreq_t *req);
#endif
