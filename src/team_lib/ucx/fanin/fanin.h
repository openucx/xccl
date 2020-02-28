#ifndef FANIN_H_
#define FANIN_H_
#include "../xccl_ucx_lib.h"

xccl_status_t xccl_ucx_fanin_linear_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_fanin_linear_progress(xccl_ucx_collreq_t *req);
#endif
