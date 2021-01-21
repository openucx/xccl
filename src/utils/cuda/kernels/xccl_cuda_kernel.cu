#include <api/xccl.h>
#include <cuda.h>

#define CUDACHECK(cmd) do {                                         \
        cudaError_t e = cmd;                                        \
        if( e != cudaSuccess && e != cudaErrorCudartUnloading ) {   \
            fprintf(stderr, "cuda failed wtih ret:%d(%s)", e,       \
                             cudaGetErrorString(e));                \
            return XCCL_ERR_NO_MESSAGE;                             \
        }                                                           \
} while(0)

__global__ void dummy_kernel(volatile xccl_status_t *status, int *is_free) {
    xccl_status_t st;

    if (*status == XCCL_OK) {
        /* was requested to stop allready */
        *is_free = 1;
        return;
    } else {
        *status = XCCL_INPROGRESS;
    }
    do {
        st = *status;
    } while(st != XCCL_OK);

    *is_free = 1;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

xccl_status_t xccl_cuda_dummy_kernel(xccl_status_t *status, int *is_free,
                                     cudaStream_t stream)
{
    dummy_kernel<<<1, 1, 0, stream>>>(status, is_free);
    CUDACHECK(cudaGetLastError());
    return XCCL_OK;
}

#ifdef __cplusplus
}
#endif
