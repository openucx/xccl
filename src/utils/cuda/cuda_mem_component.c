#include "cuda_mem_component.h"

xccl_cuda_mem_component_t xccl_cuda_mem_component;

#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            return XCCL_ERR_NO_MESSAGE;                             \
        }                                                           \
} while(0)

static xccl_status_t xccl_cuda_open()
{
    xccl_cuda_mem_component.stream = 0;
}

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
                                    size_t count, xccl_dt_t dtype, xccl_op_t op,
                                    cudaStream_t stream);

xccl_status_t xccl_cuda_reduce(void *sbuf1, void *sbuf2, void *target,
                               size_t count, xccl_dt_t dtype, xccl_op_t op)
{
    if (xccl_cuda_mem_component.stream == 0) {
        CUDACHECK(cudaStreamCreateWithFlags(&xccl_cuda_mem_component.stream,
                                            cudaStreamNonBlocking));
    }
    return xccl_cuda_reduce_impl(sbuf1, sbuf2, target, count, dtype, op,
                                 xccl_cuda_mem_component.stream);
}

xccl_status_t xccl_cuda_reduce_multi_impl(void *sbuf1, void *sbuf2, void *rbuf,
                                         size_t count, size_t size, size_t stride,
                                         xccl_dt_t dtype, xccl_op_t op,
                                         cudaStream_t stream);

xccl_status_t xccl_cuda_reduce_multi(void *sbuf1, void *sbuf2, void *rbuf,
                                     size_t count, size_t size, size_t stride,
                                     xccl_dt_t dtype, xccl_op_t op)
{
    if (xccl_cuda_mem_component.stream == 0) {
        CUDACHECK(cudaStreamCreateWithFlags(&xccl_cuda_mem_component.stream,
                                            cudaStreamNonBlocking));
    }
    return xccl_cuda_reduce_multi_impl(sbuf1, sbuf2, rbuf, count, size, stride,
                                       dtype, op,
                                       xccl_cuda_mem_component.stream);
}

xccl_status_t xccl_cuda_mem_type(void *ptr, ucs_memory_type_t *mem_type) {
    struct      cudaPointerAttributes attr;
    cudaError_t err;
    
    err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return XCCL_ERR_UNSUPPORTED;
    }

#if CUDART_VERSION >= 10000
    if (attr.type == cudaMemoryTypeDevice) {
#else
    if (attr.memoryType == cudaMemoryTypeDevice) {
#endif
        *mem_type = UCS_MEMORY_TYPE_CUDA;
    }
    else {
        *mem_type = UCS_MEMORY_TYPE_HOST;
    }

    return XCCL_OK;
}

static void xccl_cuda_close()
{
    if (xccl_cuda_mem_component.stream != 0) {
        cudaStreamDestroy(xccl_cuda_mem_component.stream);
    }
}

xccl_cuda_mem_component_t xccl_cuda_mem_component = {
    xccl_cuda_open,
    xccl_cuda_mem_alloc,
    xccl_cuda_mem_free,
    xccl_cuda_mem_type,
    xccl_cuda_reduce,
    xccl_cuda_reduce_multi,
    xccl_cuda_close
};
