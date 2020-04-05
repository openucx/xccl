#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include "mem_component.h"
#include "reduce.h"
#include "utils/xccl_log.h"

static xccl_mem_component_t *mem_components[UCS_MEMORY_TYPE_LAST];

xccl_status_t xccl_mem_component_init(const char* components_path)
{
    void   *handle;
    char   *mem_comp_path;
    size_t mem_comp_path_len;
    int    mt;

    mem_comp_path_len = strlen(components_path) + 32;
    mem_comp_path = (char*)malloc(mem_comp_path_len);

    for(mt = UCS_MEMORY_TYPE_HOST + 1; mt < UCS_MEMORY_TYPE_LAST; mt++) {
        snprintf(mem_comp_path, mem_comp_path_len, "%s/xccl_%s_mem_component.so",
                 components_path, ucs_memory_type_names[mt]);        
        handle = dlopen(mem_comp_path, RTLD_LAZY);
        if (handle) {
            mem_components[UCS_MEMORY_TYPE_CUDA] = (xccl_mem_component_t*)dlsym(handle, "xccl_cuda_mem_component");
            xccl_debug("%s mem component found", ucs_memory_type_names[mt]);
        }
    }

    free(mem_comp_path);
    return XCCL_OK;
}

xccl_status_t xccl_mem_component_alloc(void **ptr, size_t len,
                                       ucs_memory_type_t mem_type)
{
    if (mem_type == UCS_MEMORY_TYPE_HOST) {
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
                                      ucs_memory_type_t mem_type)
{
    if (mem_type == UCS_MEMORY_TYPE_HOST) {
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
                                        xccl_op_t op, ucs_memory_type_t mem_type)
{
    if (mem_type == UCS_MEMORY_TYPE_HOST) {
        return xccl_dt_reduce(sbuf1, sbuf2, target, count, dtype, op);
    }

    if (mem_components[mem_type] == NULL) {
        return XCCL_ERR_UNSUPPORTED;
    }

    return mem_components[mem_type]->reduce(sbuf1, sbuf2, target, count, dtype, op);
}

xccl_status_t xccl_mem_component_type(void *ptr, ucs_memory_type_t *mem_type)
{
    xccl_status_t st;
    int           mt;

    *mem_type = UCS_MEMORY_TYPE_HOST;

    for(mt = UCS_MEMORY_TYPE_HOST+1; mt < UCS_MEMORY_TYPE_LAST; mt++) {
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
