/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_CUDA_MEM_COMPONENT_H_
#define XCCL_CUDA_MEM_COMPONENT_H_

#include <utils/mem_component.h>
#include <cuda_runtime.h>

typedef struct xccl_cuda_mc_event {
    xccl_mc_event_t super;
    int             is_free;
    cudaEvent_t     cuda_event;
} xccl_cuda_mc_event_t;

typedef struct xccl_cuda_mem_component_stream_request {
    xccl_mem_component_stream_request_t super;
    int                                 is_free;
    void                                *dev_is_free;
    xccl_status_t                       status;
    void                                *dev_status;
    xccl_cuda_mc_event_t                *start_event;
    xccl_cuda_mc_event_t                *finish_event;
} xccl_cuda_mem_component_stream_request_t;

typedef enum xccl_cuda_mc_activity {
    XCCL_CUDA_MC_ACTIVITY_KERNEL,
    XCCL_CUDA_MC_ACTIVITY_DRIVER,
} xccl_cuda_mc_activity_t;

typedef struct xccl_cuda_mem_component {
    xccl_mem_component_t                     super;
    cudaStream_t                             stream;
    xccl_cuda_mem_component_stream_request_t *stream_requests;
    xccl_cuda_mc_event_t                     *events;
    xccl_cuda_mc_activity_t                  activity;
} xccl_cuda_mem_component_t;

#endif