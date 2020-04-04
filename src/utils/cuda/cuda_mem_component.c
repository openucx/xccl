#include "cuda_mem_component.h"

#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            return XCCL_ERR_NO_MESSAGE;                             \
        }                                                           \
} while(0)

static xccl_status_t xccl_cuda_mem_alloc(void **ptr, size_t len)
{
    CUDACHECK(cudaMalloc(ptr, len));
    return XCCL_OK;
}

static xccl_status_t xccl_cuda_mem_free(void *ptr)
{
    CUDACHECK(cudaFree(ptr));
    return XCCL_OK;
}

xccl_status_t xccl_cuda_reduce_impl(void *sbuf1, void *sbuf2, void *target,
                                    size_t count, xccl_dt_t dtype, xccl_op_t op);

xccl_status_t xccl_cuda_reduce(void *sbuf1, void *sbuf2, void *target,
                                    size_t count, xccl_dt_t dtype, xccl_op_t op)
{
    return xccl_cuda_reduce_impl(sbuf1, sbuf2, target, count, dtype, op);
}

xccl_status_t xccl_cuda_mem_type(void *ptr){
    struct cudaPointerAttributes attr;
    cudaError_t err;
    
    err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return XCCL_ERR_UNSUPPORTED;
    }

    if (attr.memoryType == cudaMemoryTypeDevice) {
        return XCCL_OK;
    }
}

xccl_cuda_mem_component_t xccl_cuda_mem_component = {
    xccl_cuda_mem_alloc,
    xccl_cuda_mem_free,
    xccl_cuda_mem_type,
    xccl_cuda_reduce,
};
