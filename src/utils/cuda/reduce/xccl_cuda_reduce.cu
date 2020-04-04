#include <api/xccl.h>

template <typename T>
__global__ void xccl_reduce_cuda_sum(T *s1, T *s2, T *target, size_t count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        target[i] = s1[i] + s2[i];
    }
}

#ifdef __cplusplus
extern "C" {
#endif

xccl_status_t xccl_cuda_reduce_impl(void *sbuf1, void *sbuf2, void *target,
                                    size_t count, xccl_dt_t dtype, xccl_op_t op)
{
    if (op == XCCL_OP_SUM) {
        switch (dtype)
        {
            case XCCL_DT_INT32:
                xccl_reduce_cuda_sum<float><<<(count + 255)/256, 256>>>((float*)sbuf1, (float*)sbuf2, (float*)target, count);
                return XCCL_OK;
            default:
                return XCCL_ERR_NOT_IMPLEMENTED;
        }

    }
    return XCCL_ERR_NOT_IMPLEMENTED;
}

#ifdef __cplusplus
}
#endif
