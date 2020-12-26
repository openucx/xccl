/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_CUDA_MEM_COMPONENT_H_
#define XCCL_CUDA_MEM_COMPONENT_H_

#include <utils/mem_component.h>
#include <cuda_runtime.h>

typedef struct xccl_cuda_mem_component_stream_request {
    xccl_mem_component_stream_request_t super;
    int         is_free;
    int         stop_request;
    void        *dev_stop_request;
    cudaEvent_t event;
} xccl_cuda_mem_component_stream_request_t;

typedef struct xccl_cuda_mem_component {
    xccl_mem_component_t                     super;
    cudaStream_t                             stream;
    xccl_cuda_mem_component_stream_request_t *stream_requests;
} xccl_cuda_mem_component_t;

#endif