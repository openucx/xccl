#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_
#include "../xccl_ucx_lib.h"

xccl_status_t xccl_ucx_allreduce_knomial_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_allreduce_knomial_progress(xccl_ucx_collreq_t *req);

xccl_status_t xccl_ucx_allreduce_sra_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_allreduce_sra_progress(xccl_ucx_collreq_t *req);

#endif
