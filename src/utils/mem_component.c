#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "mem_component.h"
#include "reduce.h"

static xccl_mem_component_t *mem_components[XCCL_MEMORY_TYPE_LAST];

xccl_status_t xccl_mem_component_init(const char* components_path)
{
    void   *handle;
    char   *mem_comp_path;
    size_t mem_comp_path_len;

    mem_comp_path_len = strlen(components_path) + 32;
    mem_comp_path = (char*)malloc(mem_comp_path_len);
    /* TODO fix hardcoded values */
    snprintf(mem_comp_path, mem_comp_path_len, "%s/%s", components_path, 
             "xccl_cuda_mem_component.so");
    handle = dlopen(mem_comp_path, RTLD_LAZY);
    if (handle) {
        mem_components[XCCL_MEMORY_TYPE_CUDA] = (xccl_mem_component_t*)dlsym(handle, "xccl_cuda_mem_component");
        fprintf(stderr, "CUDA mem component found");
    }

    free(mem_comp_path);
    return XCCL_OK;
}

xccl_status_t xccl_mem_component_alloc(void **ptr, size_t len,
                                       xccl_memory_type_t mem_type)
{
    if (mem_type == XCCL_MEMORY_TYPE_HOST) {
        *ptr = malloc(len);
        if (!(*ptr)) {
            return XCCL_ERR_NO_MEMORY;
        }
        return XCCL_OK;
    }

    if (mem_components[mem_type] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    return mem_components[mem_type]->mem_alloc(ptr, len);
}

xccl_status_t xccl_mem_component_free(void *ptr,
                                      xccl_memory_type_t mem_type)
{
    if (mem_type == XCCL_MEMORY_TYPE_HOST) {
        free(ptr);
        return XCCL_OK;
    }

    if (mem_components[mem_type] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    return mem_components[mem_type]->mem_free(ptr);
}

xccl_status_t xccl_mem_component_reduce(void *sbuf1, void *sbuf2, void *target,
                                        size_t count, xccl_dt_t dtype,
                                        xccl_op_t op, xccl_memory_type_t mem_type)
{
    if (mem_type == XCCL_MEMORY_TYPE_HOST) {
        return xccl_dt_reduce(sbuf1, sbuf2, target, count, dtype, op);
    }

    if (mem_components[mem_type] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    return mem_components[mem_type]->reduce(sbuf1, sbuf2, target, count, dtype, op);
}

xccl_status_t xccl_mem_component_type(void *ptr, xccl_memory_type_t *mem_type)
{
    xccl_status_t st;
    int           mt;

    *mem_type = XCCL_MEMORY_TYPE_HOST;

    for(mt = XCCL_MEMORY_TYPE_HOST+1; mt < XCCL_MEMORY_TYPE_LAST; mt++) {
        if (mem_components[mt] != NULL) {
            st = mem_components[mt]->mem_type(ptr);
            if (st == XCCL_OK) {
                *mem_type = mt;
                return XCCL_OK;
            }
        }
    }

    return XCCL_OK;
}
