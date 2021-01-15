#include <api/xccl.h>
#include <cuda.h>

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

cudaError_t xccl_cuda_dummy_kernel(volatile xccl_status_t *status, int *is_free,
                                   cudaStream_t stream)
{
    dummy_kernel<<<1, 1, 0, stream>>>(status, is_free);
    return cudaGetLastError();
}

#ifdef __cplusplus
}
#endif
