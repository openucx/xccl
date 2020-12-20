#include "cuda_mem_component.h"

xccl_cuda_mem_component_t xccl_cuda_mem_component;

#define NUM_STREAM_REQUESTS 4

#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            return XCCL_ERR_NO_MESSAGE;                             \
        }                                                           \
} while(0)

static xccl_status_t xccl_cuda_open()
{
    xccl_cuda_mem_component.stream = 0;
    return XCCL_OK;
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

static xccl_status_t xccl_cuda_alloc_resources()
{
    int i;

    CUDACHECK(cudaStreamCreateWithFlags(&xccl_cuda_mem_component.stream,
                                        cudaStreamNonBlocking));
    CUDACHECK(cudaHostAlloc((void**)&xccl_cuda_mem_component.stream_requests,
              NUM_STREAM_REQUESTS * sizeof(xccl_cuda_mem_component_stream_request_t),
              cudaHostAllocMapped));
    for (i = 0; i < NUM_STREAM_REQUESTS; i++) {
        xccl_cuda_mem_component.stream_requests[i].is_free = 1;
        CUDACHECK(cudaHostGetDevicePointer(
                  (void**)&(xccl_cuda_mem_component.stream_requests[i].dev_stop_request),
                  (void*)&(xccl_cuda_mem_component.stream_requests[i].stop_request),
                  0));
        CUDACHECK(cudaEventCreateWithFlags(
                  &xccl_cuda_mem_component.stream_requests[i].event,
                  cudaEventDisableTiming));
    }

    return XCCL_OK;
}

static xccl_status_t
xccl_cuda_get_free_stream_request(xccl_cuda_mem_component_stream_request_t **request) {
    int i;
    xccl_cuda_mem_component_stream_request_t *req;
    for (i = 0; i < NUM_STREAM_REQUESTS; i++) {
        req = xccl_cuda_mem_component.stream_requests + i;
        if (req->is_free) {
            req->is_free = 0;
            *request = req;
            return XCCL_OK;
        }
    }

    return XCCL_ERR_NO_RESOURCE;
}

xccl_status_t xccl_cuda_reduce_impl(void *sbuf1, void *sbuf2, void *target,
                                    size_t count, xccl_dt_t dtype, xccl_op_t op,
                                    cudaStream_t stream);

xccl_status_t xccl_cuda_reduce(void *sbuf1, void *sbuf2, void *target,
                               size_t count, xccl_dt_t dtype, xccl_op_t op)
{
    xccl_status_t st;

    if (xccl_cuda_mem_component.stream == 0) {
        st = xccl_cuda_alloc_resources();
        if (st != XCCL_OK) {
            return st;
        }
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
    xccl_status_t st;

    if (xccl_cuda_mem_component.stream == 0) {
        st = xccl_cuda_alloc_resources();
        if (st != XCCL_OK) {
            return st;
        }
    }
    return xccl_cuda_reduce_multi_impl(sbuf1, sbuf2, rbuf, count, size, stride,
                                       dtype, op,
                                       xccl_cuda_mem_component.stream);
}

cudaError_t xccl_cuda_dummy_kernel(int *stop, cudaStream_t stream);

xccl_status_t
xccl_cuda_start_acitivity(xccl_stream_t *stream,
                          xccl_mem_component_stream_request_t **req)
{
    xccl_cuda_mem_component_stream_request_t *request;
    int *dev_stop_request;
    xccl_status_t st;
    cudaStream_t internal_stream, user_stream;

    if (xccl_cuda_mem_component.stream == 0) {
        st = xccl_cuda_alloc_resources();
        if (st != XCCL_OK) {
            return st;
        }
    }

    st = xccl_cuda_get_free_stream_request(&request);
    if (st != XCCL_OK) {
        return st;
    }

    request->stop_request = 0;
    user_stream = *((cudaStream_t*)stream->stream);
    internal_stream = xccl_cuda_mem_component.stream;
    CUDACHECK(xccl_cuda_dummy_kernel(request->dev_stop_request, internal_stream));
    CUDACHECK(cudaEventRecord(request->event, internal_stream));
    CUDACHECK(cudaStreamWaitEvent(user_stream, request->event, 0));
    *req = &request->super;

    return XCCL_OK;
}

xccl_status_t
xccl_cuda_finish_acitivity(xccl_mem_component_stream_request_t *req)
{
    xccl_cuda_mem_component_stream_request_t *request;

    request = (xccl_cuda_mem_component_stream_request_t*)req;
    request->stop_request = 1;
    request->is_free = 1;

    return XCCL_OK;
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
    int i;

    if (xccl_cuda_mem_component.stream != 0) {
        for (i = 0; i < NUM_STREAM_REQUESTS; i++) {
            cudaEventDestroy(xccl_cuda_mem_component.stream_requests[i].event);
        }
        cudaFreeHost(xccl_cuda_mem_component.stream_requests);
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
    xccl_cuda_start_acitivity,
    xccl_cuda_finish_acitivity,
    xccl_cuda_close
};
