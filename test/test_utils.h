#include <api/xccl.h>

typedef enum test_mem_type {
    TEST_MEM_TYPE_HOST,
    TEST_MEM_TYPE_CUDA
} test_mem_type_t;

typedef enum test_memcpy_kind {
    TEST_MEMCPY_H2H,
    TEST_MEMCPY_H2D,
    TEST_MEMCPY_D2H,
    TEST_MEMCPY_D2D
} test_memcpy_kind_t;

xccl_status_t test_xccl_set_device(test_mem_type_t mtype);

xccl_status_t test_xccl_mem_alloc(void **ptr, size_t size, test_mem_type_t mtype);

xccl_status_t test_xccl_mem_free(void *ptr, test_mem_type_t mtype);

xccl_status_t test_xccl_memcpy(void *dst, void *src, size_t size, test_memcpy_kind_t kind);

xccl_status_t test_xccl_memset(void *ptr, int value, size_t size, test_mem_type_t mtype);

xccl_status_t test_xccl_memcmp(void *ptr1, test_mem_type_t ptr1_mtype,
                               void *ptr2, test_mem_type_t ptr2_mtype,
                               size_t size, int *result);
