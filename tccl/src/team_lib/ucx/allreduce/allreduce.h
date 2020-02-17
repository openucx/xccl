#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_
#include "../tccl_ucx_lib.h"

tccl_status_t tccl_ucx_allreduce_knomial_start(tccl_ucx_collreq_t *req);
tccl_status_t tccl_ucx_allreduce_knomial_progress(tccl_ucx_collreq_t *req);
#endif
