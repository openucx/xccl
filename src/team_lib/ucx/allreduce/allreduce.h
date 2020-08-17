#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_

#include "../xccl_ucx_lib.h"

typedef enum {
    XCCL_UCX_ALLREDUCE_ALG_KNOMIAL,
    XCCL_UCX_ALLREDUCE_ALG_SRA,
    XCCL_UCX_ALLREDUCE_ALG_AUTO,
    XCCL_UCX_ALLREDUCE_ALG_LAST
} xccl_ucx_allreduce_alg_t;

extern const xccl_ucx_coll_start_fn_p xccl_ucx_allreduce_start[];
extern const char* xccl_allreduce_alg_names[];

xccl_status_t xccl_ucx_allreduce_knomial_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_allreduce_knomial_progress(xccl_ucx_collreq_t *req);

xccl_status_t xccl_ucx_allreduce_sra_start(xccl_ucx_collreq_t *req);
xccl_status_t xccl_ucx_allreduce_sra_progress(xccl_ucx_collreq_t *req);

#endif
