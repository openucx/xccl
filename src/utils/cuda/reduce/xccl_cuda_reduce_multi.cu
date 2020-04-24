#include <api/xccl.h>
#include <assert.h>
#include <stdio.h>


#define  DO_OP_MAX(_v1, _v2)  (_v1 > _v2 ? _v1 : _v2)
#define  DO_OP_MIN(_v1, _v2)  (_v1 < _v2 ? _v1 : _v2)
#define  DO_OP_SUM(_v1, _v2)  (_v1 + _v2)
#define DO_OP_PROD(_v1, _v2)  (_v1 * _v2)
#define DO_OP_LAND(_v1, _v2)  (_v1 && _v2)
#define DO_OP_BAND(_v1, _v2)  (_v1 & _v2)
#define  DO_OP_LOR(_v1, _v2)  (_v1 || _v2)
#define  DO_OP_BOR(_v1, _v2)  (_v1 | _v2)
#define DO_OP_LXOR(_v1, _v2)  ((!_v1) != (!_v2))
#define DO_OP_BXOR(_v1, _v2)  (_v1 ^ _v2)

#define DO_CUDA_REDUCE_MULTI_WITH_OP(NAME, OP)                                     \
template <typename T>                                                              \
__global__ void XCCL_REDUCE_MULTI_CUDA_ ## NAME (T *s1, T *s2, T *t, size_t count, \
                                                 size_t size, size_t stride)       \
{                                                                                  \
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;                          \
        if (i<size) {                                                              \
            t[i] = OP(s1[i], s2[i]);                                               \
            for(size_t j = 1; j < count; j++) {                                    \
                t[i] = OP(t[i], s2[i + j*stride]);                                 \
            }                                                                      \
        }                                                                          \
}                                                                                  \

DO_CUDA_REDUCE_MULTI_WITH_OP(MAX,  DO_OP_MAX)
DO_CUDA_REDUCE_MULTI_WITH_OP(MIN,  DO_OP_MIN)
DO_CUDA_REDUCE_MULTI_WITH_OP(SUM,  DO_OP_SUM)
DO_CUDA_REDUCE_MULTI_WITH_OP(PROD, DO_OP_PROD)
DO_CUDA_REDUCE_MULTI_WITH_OP(LAND, DO_OP_LAND)
DO_CUDA_REDUCE_MULTI_WITH_OP(BAND, DO_OP_BAND)
DO_CUDA_REDUCE_MULTI_WITH_OP(LOR,  DO_OP_LOR)
DO_CUDA_REDUCE_MULTI_WITH_OP(BOR,  DO_OP_BOR)
DO_CUDA_REDUCE_MULTI_WITH_OP(LXOR, DO_OP_LXOR)
DO_CUDA_REDUCE_MULTI_WITH_OP(BXOR, DO_OP_BXOR)

#define DO_LAUNCH_KERNEL(NAME, type, src1, src2, dest, count, size, stride, stream) do { \
        XCCL_REDUCE_MULTI_CUDA_ ## NAME<type> <<<(count + 255)/256, 256, 0, stream>>>(src1, src2, dest, count, size, stride); \
    } while(0)

#define DO_DT_REDUCE_INT(type, op, src1_p, src2_p, dest_p, count, size, stride, stream) do { \
        type *src1 = (type *)src1_p;                                                         \
        type *src2 = (type *)src2_p;                                                         \
        type *dest = (type *)dest_p;                                                         \
        switch(op) {                                                                         \
        case XCCL_OP_MAX:                                                                    \
            DO_LAUNCH_KERNEL(MAX, type, src1, src2, dest, count, size, stride, stream);      \
            break;                                                                           \
        case XCCL_OP_MIN:                                                                    \
            DO_LAUNCH_KERNEL(MIN, type, src1, src2, dest, count, size, stride, stream);      \
            break;                                                                           \
        case XCCL_OP_SUM:                                                                    \
            DO_LAUNCH_KERNEL(SUM, type, src1, src2, dest, count, size, stride, stream);      \
            break;                                                                           \
        case XCCL_OP_PROD:                                                                   \
            DO_LAUNCH_KERNEL(PROD, type, src1, src2, dest, count, size, stride, stream);     \
            break;                                                                           \
        case XCCL_OP_LAND:                                                                   \
            DO_LAUNCH_KERNEL(LAND, type, src1, src2, dest, count, size, stride, stream);     \
            break;                                                                           \
        case XCCL_OP_BAND:                                                                   \
            DO_LAUNCH_KERNEL(BAND, type, src1, src2, dest, count, size, stride, stream);     \
            break;                                                                           \
        case XCCL_OP_LOR:                                                                    \
            DO_LAUNCH_KERNEL(LOR, type, src1, src2, dest, count, size, stride, stream);      \
            break;                                                                           \
        case XCCL_OP_LXOR:                                                                   \
            DO_LAUNCH_KERNEL(LXOR, type, src1, src2, dest, count, size, stride, stream);     \
            break;                                                                           \
        case XCCL_OP_BXOR:                                                                   \
            DO_LAUNCH_KERNEL(BXOR, type, src1, src2, dest, count, size, stride, stream);     \
            break;                                                                           \
        default:                                                                             \
            return XCCL_ERR_UNSUPPORTED;                                                     \
        }                                                                                    \
    } while(0)

#define DO_DT_REDUCE_FLOAT(type, op, src1_p, src2_p, dest_p, count, size, stride, stream) do { \
        type *src1 = (type *)src1_p;                                                           \
        type *src2 = (type *)src2_p;                                                           \
        type *dest = (type *)dest_p;                                                           \
        switch(op) {                                                                           \
        case XCCL_OP_MAX:                                                                      \
            DO_LAUNCH_KERNEL(MAX, type, src1, src2, dest, count, size, stride, stream);        \
            break;                                                                             \
        case XCCL_OP_MIN:                                                                      \
            DO_LAUNCH_KERNEL(MIN, type, src1, src2, dest, count, size, stride, stream);        \
            break;                                                                             \
        case XCCL_OP_SUM:                                                                      \
            DO_LAUNCH_KERNEL(SUM, type, src1, src2, dest, count, size, stride, stream);        \
            break;                                                                             \
        case XCCL_OP_PROD:                                                                     \
            DO_LAUNCH_KERNEL(PROD, type, src1, src2, dest, count, size, stride, stream);       \
            break;                                                                             \
        default:                                                                               \
            return XCCL_ERR_UNSUPPORTED;                                                       \
        }                                                                                      \
    } while(0)


#ifdef __cplusplus
extern "C" {
#endif

xccl_status_t xccl_cuda_reduce_multi_impl(void *sbuf1, void *sbuf2, void *rbuf,
                                          size_t count, size_t size, size_t stride,
                                          xccl_dt_t dtype, xccl_op_t op,
                                          cudaStream_t stream)
{
    size_t ld = stride/xccl_dt_size(dtype);

    assert((stride % xccl_dt_size(dtype) == 0));
    switch (dtype)
    {
        case XCCL_DT_INT16:
            DO_DT_REDUCE_INT(int16_t, op, sbuf1, sbuf2, rbuf, count, size, ld, stream);
            break;     
        case XCCL_DT_INT32:
            DO_DT_REDUCE_INT(int32_t, op, sbuf1, sbuf2, rbuf, count, size, ld, stream);
            break;
        case XCCL_DT_INT64:
            DO_DT_REDUCE_INT(int64_t, op, sbuf1, sbuf2, rbuf, count, size, ld, stream);
            break;
        case XCCL_DT_FLOAT32:
            assert(4 == sizeof(float));
            DO_DT_REDUCE_FLOAT(float, op, sbuf1, sbuf2, rbuf, count, size, ld, stream);
            break;
        case XCCL_DT_FLOAT64:
            assert(8 == sizeof(double));
            DO_DT_REDUCE_FLOAT(double, op, sbuf1, sbuf2, rbuf, count, size, ld, stream);
            break;
        default:
            return XCCL_ERR_UNSUPPORTED;
    }
    cudaStreamSynchronize(stream);
    return XCCL_OK;
}

#ifdef __cplusplus
}
#endif
