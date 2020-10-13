#include "test_utils.h"

#include <stdlib.h>
#include <stdio.h>

#ifdef XCCL_TEST_WITH_CUDA
#include <cuda_runtime.h>

static const test_memcpy_kind_t test_memcpy_kind_to_cuda[] = {
    [TEST_MEMCPY_H2H] = cudaMemcpyHostToHost,
    [TEST_MEMCPY_H2D] = cudaMemcpyHostToDevice,
    [TEST_MEMCPY_D2H] = cudaMemcpyDeviceToHost,
    [TEST_MEMCPY_D2D] = cudaMemcpyDeviceToDevice,
};

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if(cudaSuccess != e) {                            \
    fprintf(stderr, "CUDA error %s:%d '%d' %s\n",   \
        __FILE__,__LINE__, e, cudaGetErrorName(e)); \
    return XCCL_ERR_NO_MESSAGE;                     \
  }                                                 \
} while(0)

#endif

xccl_status_t test_xccl_set_device(test_mem_type_t mtype)
{
    char *local_rank;

    if (mtype == TEST_MEM_TYPE_CUDA) {
#ifdef XCCL_TEST_WITH_CUDA
        local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
        if (local_rank) {
            CUDACHECK(cudaSetDevice(atoi(local_rank)) != cudaSuccess);
        }
#else
        fprintf(stderr, "test wasn't compiled with CUDA support\n");
        return XCCL_ERR_INVALID_PARAM;
#endif
    }
    return XCCL_OK;
}

xccl_status_t test_xccl_mem_alloc(void **ptr, size_t size, test_mem_type_t mtype)
{
    switch (mtype) {
        case TEST_MEM_TYPE_HOST:
            *ptr = malloc(size);
            if (*ptr == NULL) {
                return XCCL_ERR_NO_MEMORY;
            }
            return XCCL_OK;
        case TEST_MEM_TYPE_CUDA:
#ifdef XCCL_TEST_WITH_CUDA
            CUDACHECK(cudaMalloc(ptr, size));
            return XCCL_OK;
#else
            fprintf(stderr, "test wasn't compiled with CUDA support\n");
            return XCCL_ERR_INVALID_PARAM;
#endif
        default:
            return XCCL_ERR_INVALID_PARAM;
    }
}

xccl_status_t test_xccl_mem_free(void *ptr, test_mem_type_t mtype)
{
    switch (mtype) {
        case TEST_MEM_TYPE_HOST:
            free(ptr);
            return XCCL_OK;
        case TEST_MEM_TYPE_CUDA:
#ifdef XCCL_TEST_WITH_CUDA
            CUDACHECK(cudaFree(ptr));
            return XCCL_OK;
#else
            fprintf(stderr, "test wasn't compiled with CUDA support\n");
            return XCCL_ERR_INVALID_PARAM;
#endif
        default:
            return XCCL_ERR_INVALID_PARAM;
    }
}

xccl_status_t test_xccl_memcpy(void *dst, void *src, size_t size, test_memcpy_kind_t kind)
{
    switch(kind) {
        case TEST_MEMCPY_H2H:
            memcpy(dst, src, size);
            return XCCL_OK;
        case TEST_MEMCPY_H2D: case TEST_MEMCPY_D2H: case TEST_MEMCPY_D2D:
#ifdef XCCL_TEST_WITH_CUDA
            CUDACHECK(cudaMemcpy(dst, src, size, test_memcpy_kind_to_cuda[kind]));
            return XCCL_OK;
#else
            fprintf(stderr, "test wasn't compiled with CUDA support\n");
            return XCCL_ERR_INVALID_PARAM;
#endif
        default:
            return XCCL_ERR_INVALID_PARAM;
    }
}

xccl_status_t test_xccl_memset(void *ptr, int value, size_t size,
                               test_mem_type_t mtype)
{
    switch (mtype) {
        case TEST_MEM_TYPE_HOST:
            memset(ptr, value, size);
            return XCCL_OK;
        case TEST_MEM_TYPE_CUDA:
#ifdef XCCL_TEST_WITH_CUDA
            CUDACHECK(cudaMemset(ptr, value, size));
            return XCCL_OK;
#else
            fprintf(stderr, "test wasn't compiled with CUDA support\n");
            return XCCL_ERR_INVALID_PARAM;
#endif
        default:
            return XCCL_ERR_INVALID_PARAM;
    }
}

xccl_status_t test_xccl_memcmp(void *ptr1, test_mem_type_t ptr1_mtype,
                               void *ptr2, test_mem_type_t ptr2_mtype,
                               size_t size, int *result)
{
    void *ptr1_host, *ptr2_host;

    if (ptr1_mtype == TEST_MEM_TYPE_HOST) {
        ptr1_host = ptr1;
    }
    else {
        ptr1_host = malloc(size);
        if (ptr1_host == NULL) {
            return XCCL_ERR_NO_MEMORY;
        }
        if (test_xccl_memcpy(ptr1_host, ptr1, size, TEST_MEMCPY_D2H) != XCCL_OK) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }

    if (ptr2_mtype == TEST_MEM_TYPE_HOST) {
        ptr2_host = ptr2;
    }
    else {
        ptr2_host = malloc(size);
        if (ptr2_host == NULL) {
            return XCCL_ERR_NO_MEMORY;
        }
        if (test_xccl_memcpy(ptr2_host, ptr2, size, TEST_MEMCPY_D2H) != XCCL_OK) {
            return XCCL_ERR_NO_MESSAGE;
        }
    }

    *result = memcmp(ptr1_host, ptr2_host, size);
    if (ptr1_mtype != TEST_MEM_TYPE_HOST) {
        free(ptr1_host);
    }
    if (ptr2_mtype != TEST_MEM_TYPE_HOST) {
        free(ptr2_host);
    }

    return XCCL_OK;
}
