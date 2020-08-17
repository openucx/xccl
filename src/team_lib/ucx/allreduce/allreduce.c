#include "allreduce.h"

const char* xccl_allreduce_alg_names[] = {
    [XCCL_UCX_ALLREDUCE_ALG_KNOMIAL] = "knomial",
    [XCCL_UCX_ALLREDUCE_ALG_SRA]     = "sra",
    [XCCL_UCX_ALLREDUCE_ALG_AUTO]    = "auto",
};

const xccl_ucx_coll_start_fn_p xccl_ucx_allreduce_start[] = {
    [XCCL_UCX_ALLREDUCE_ALG_KNOMIAL] = xccl_ucx_allreduce_knomial_start,
    [XCCL_UCX_ALLREDUCE_ALG_SRA]     = xccl_ucx_allreduce_sra_start
};
