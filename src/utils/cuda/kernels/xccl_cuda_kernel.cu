#include <api/xccl.h>
#include <cuda.h>

__global__ void dummy_kernel(volatile int *stop) {
    int should_stop;
    do {
        should_stop = *stop;
    } while(!should_stop);
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t xccl_cuda_dummy_kernel(int *stop, cudaStream_t stream)
{
    dummy_kernel<<<1, 1, 0, stream>>>(stop);
    return cudaGetLastError();
}

#ifdef __cplusplus
}
#endif
