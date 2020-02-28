#ifndef BARRIER_H_
#define BARRIER_H_
#include "../xccl_ucx_lib.h"

xccl_status_t xccl_ucx_barrier_knomial_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_barrier_knomial_progress(xccl_ucx_collreq_t *req);
#endif
