#include "cuda_mem_component.h"
#include <stdlib.h>
#include <cuda.h>

xccl_cuda_mem_component_t xccl_cuda_mem_component;

#define NUM_STREAM_REQUESTS 128
#define NUM_EVENTS (2*NUM_STREAM_REQUESTS)

#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            fprintf(stderr, "cuda failed wtih ret:%d(%s)", e,       \
                             cudaGetErrorString(e));                \
            return XCCL_ERR_NO_MESSAGE;                             \
        }                                                           \
} while(0)

#define CUCHECK(cmd) do {                                  \
        CUresult e = cmd;                                  \
        if( e != CUDA_SUCCESS) {                           \
            fprintf(stderr, "cuda failed wtih ret:%d", e); \
            return XCCL_ERR_NO_MESSAGE;                    \
        }                                                  \
} while(0)

#define XCCL_CUDA_INIT_RESOUCES() do {               \
    if (xccl_cuda_mem_component.stream == 0) {       \
        xccl_status_t st_alloc_resc;                 \
        st_alloc_resc = xccl_cuda_alloc_resources(); \
        if (st_alloc_resc != XCCL_OK) {              \
            return st_alloc_resc;                    \
        }                                            \
    }                                                \
} while(0)                                           \

static xccl_status_t xccl_cuda_open()
{
    //TODO: use parser
    char *env;

    xccl_cuda_mem_component.stream = 0;
    xccl_cuda_mem_component.activity = XCCL_CUDA_MC_ACTIVITY_KERNEL;
    env = getenv("XCCL_MC_CUDA_ACTIVITY");
    if (env) {
        if (strcmp(env, "kernel") == 0) {
            xccl_cuda_mem_component.activity_fn_type = XCCL_CUDA_MC_ACTIVITY_KERNEL;
        } else if (strcmp(env, "driver") == 0) {
            xccl_cuda_mem_component.activity_fn_type = XCCL_CUDA_MC_ACTIVITY_DRIVER;
        }
    }
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

xccl_status_t xccl_cuda_dummy_kernel(xccl_status_t *status, int *is_free,
                                     cudaStream_t stream);

xccl_status_t xccl_activity_driver(xccl_status_t *status, int *is_free,
                                   cudaStream_t stream)
{
    CUdeviceptr status_ptr  = (CUdeviceptr)status;
    CUdeviceptr is_free_ptr = (CUdeviceptr)is_free;

    CUCHECK(cuStreamWriteValue32(stream, status_ptr, XCCL_INPROGRESS, 0));
    CUCHECK(cuStreamWaitValue32(stream, status_ptr, XCCL_OK,
                                CU_STREAM_WAIT_VALUE_EQ));
    CUCHECK(cuStreamWriteValue32(stream, is_free_ptr, 1, 0));
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
    xccl_cuda_mem_component.events = (xccl_cuda_mc_event_t*)malloc(
                                       NUM_EVENTS*sizeof(xccl_cuda_mc_event_t));
    for (i = 0; i < NUM_STREAM_REQUESTS; i++) {
        xccl_cuda_mem_component.stream_requests[i].is_free = 1;
        CUDACHECK(cudaHostGetDevicePointer(
                  (void**)&(xccl_cuda_mem_component.stream_requests[i].dev_status),
                  (void*)&(xccl_cuda_mem_component.stream_requests[i].status),
                  0));
        CUDACHECK(cudaHostGetDevicePointer(
                  (void**)&(xccl_cuda_mem_component.stream_requests[i].dev_is_free),
                  (void*)&(xccl_cuda_mem_component.stream_requests[i].is_free),
                  0));
    }
    for (i = 0; i < NUM_EVENTS; i++) {
        xccl_cuda_mem_component.events[i].is_free = 1;
        CUDACHECK(cudaEventCreateWithFlags(
                  &xccl_cuda_mem_component.events[i].cuda_event,
                  cudaEventDisableTiming));
    }
    if (xccl_cuda_mem_component.activity_fn_type == XCCL_CUDA_MC_ACTIVITY_DRIVER) {
        CUdevice device;
        int attr;

        CUCHECK(cuCtxGetDevice(&device));
        CUCHECK(cuDeviceGetAttribute(&attr,
                                     CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                                     device));
        if (attr == 0) {
            printf("CUDA MC: cuda memops are not supported or disabled "
                   "fallback to kernel\n");
            xccl_cuda_mem_component.activity = xccl_cuda_dummy_kernel;
        } else {
            xccl_cuda_mem_component.activity = xccl_activity_driver;
        }
    } else {
        xccl_cuda_mem_component.activity = xccl_cuda_dummy_kernel;
    }

    return XCCL_OK;
}

/* Consider using UCS allocator */
static xccl_status_t
xccl_cuda_get_free_event(xccl_cuda_mc_event_t **event) {
    int i;
    xccl_cuda_mc_event_t *et;
    for (i = 0; i < NUM_EVENTS; i++) {
        et = xccl_cuda_mem_component.events + i;
        if (et->is_free) {
            et->is_free = 0;
            *event = et;
            return XCCL_OK;
        }
    }

    return XCCL_ERR_NO_RESOURCE;
}

static xccl_status_t
xccl_cuda_get_free_stream_request(
    xccl_cuda_mem_component_stream_request_t **request)
{
    int i;
    xccl_status_t status;
    xccl_cuda_mem_component_stream_request_t *req;
    for (i = 0; i < NUM_STREAM_REQUESTS; i++) {
        req = xccl_cuda_mem_component.stream_requests + i;
        if (req->is_free) {
            status = xccl_cuda_get_free_event(&req->start_event);
            if (status != XCCL_OK) {
               return status;
            }
            status = xccl_cuda_get_free_event(&req->finish_event);
            if (status != XCCL_OK) {
                req->start_event->is_free = 1;
                return status;
            }
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
    XCCL_CUDA_INIT_RESOUCES();
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
    XCCL_CUDA_INIT_RESOUCES();
    return xccl_cuda_reduce_multi_impl(sbuf1, sbuf2, rbuf, count, size, stride,
                                       dtype, op,
                                       xccl_cuda_mem_component.stream);
}

xccl_status_t
xccl_cuda_start_acitivity(xccl_stream_t *stream,
                          xccl_mem_component_stream_request_t **req)
{
    xccl_cuda_mem_component_stream_request_t *request;
    xccl_status_t st;
    cudaStream_t internal_stream, user_stream;

    XCCL_CUDA_INIT_RESOUCES();
    st = xccl_cuda_get_free_stream_request(&request);
    if (st != XCCL_OK) {
        fprintf(stderr, "cuda mc: failed to get stream req (%d)\n", st);
        return st;
    }

    request->status = XCCL_INITIALIZED;
    user_stream = *((cudaStream_t*)stream->stream);
    internal_stream = xccl_cuda_mem_component.stream;
    CUDACHECK(cudaEventRecord(request->start_event->cuda_event, user_stream));
    CUDACHECK(cudaStreamWaitEvent(internal_stream,
                                  request->start_event->cuda_event, 0));
    st = xccl_cuda_mem_component.activity(request->dev_status,
                                          request->dev_is_free,
                                          internal_stream);
    if (st != XCCL_OK) {
        return st;
    }
    CUDACHECK(cudaEventRecord(request->finish_event->cuda_event, internal_stream));
    CUDACHECK(cudaStreamWaitEvent(user_stream,
                                  request->finish_event->cuda_event, 0));
    *req = &request->super;

    return XCCL_OK;
}
xccl_status_t
xccl_cuda_query_acitivity(xccl_mem_component_stream_request_t *req)
{

    xccl_cuda_mem_component_stream_request_t *request;

    request = ucs_derived_of(req, xccl_cuda_mem_component_stream_request_t);
    return request->status;
}


xccl_status_t
xccl_cuda_finish_acitivity(xccl_mem_component_stream_request_t *req)
{
    xccl_cuda_mem_component_stream_request_t *request;

    request = ucs_derived_of(req, xccl_cuda_mem_component_stream_request_t);
    /* set status to XCCL_OK to request kernel to stop */
    request->start_event->is_free = 1;
    request->finish_event->is_free = 1;
    request->status = XCCL_OK;

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

xccl_status_t xccl_cuda_event_record(xccl_stream_t *stream,
                                     xccl_mc_event_t **event)
{
    xccl_cuda_mc_event_t *et;
    cudaStream_t user_stream;
    xccl_status_t st;

    XCCL_CUDA_INIT_RESOUCES();
    st = xccl_cuda_get_free_event(&et);
    if (st != XCCL_OK) {
        fprintf(stderr, "cuda mc: failed to get free event (%d)\n", st);
        return st;
    }

    user_stream = *((cudaStream_t*)stream->stream);
    CUDACHECK(cudaEventRecord(et->cuda_event, user_stream));

    *event = &et->super;
    return XCCL_OK;
}

xccl_status_t xccl_cuda_event_query(xccl_mc_event_t *event)
{
    xccl_cuda_mc_event_t *et = ucs_derived_of(event, xccl_cuda_mc_event_t);
    cudaError_t cuda_st;

    cuda_st = cudaEventQuery(et->cuda_event);
    switch(cuda_st) {
    case cudaSuccess:
        return XCCL_OK;
    case cudaErrorNotReady:
        return XCCL_INPROGRESS;
    default:
        return XCCL_ERR_NO_MESSAGE;
    }
}

xccl_status_t xccl_cuda_event_free(xccl_mc_event_t *event)
{
    xccl_cuda_mc_event_t *et = ucs_derived_of(event, xccl_cuda_mc_event_t);

    et->is_free = 1;
    return XCCL_OK;
}

static void xccl_cuda_close()
{
    int i;

    if (xccl_cuda_mem_component.stream != 0) {
        for (i = 0; i < NUM_EVENTS; i++) {
            cudaEventDestroy(xccl_cuda_mem_component.events[i].cuda_event);
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
    xccl_cuda_event_record,
    xccl_cuda_event_query,
    xccl_cuda_event_free,
    xccl_cuda_start_acitivity,
    xccl_cuda_query_acitivity,
    xccl_cuda_finish_acitivity,
    xccl_cuda_close
};
