#ifndef FANIN_H_
#define FANIN_H_
#include "../tccl_ucx_lib.h"

tccl_status_t tccl_ucx_fanin_linear_start(tccl_ucx_collreq_t *req);
tccl_status_t tccl_ucx_fanin_linear_progress(tccl_ucx_collreq_t *req);
#endif
