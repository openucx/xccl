#pragma once

#include <ucp/api/ucp.h>

#define TORCH_UCX_ANY_SOURCE                   -1
#define TORCH_UCX_TAG_BITS                     33
#define TORCH_UCX_RANK_BITS                    31
#define TORCH_XCCL_OOB_TAG                     UCS_BIT(TORCH_UCX_TAG_BITS)
#define TORCH_UCX_ANY_SOURCE_MASK              0xfffffffe00000000ul
#define TORCH_UCX_SPECIFIC_SOURCE_MASK         0xfffffffffffffffful

#define TORCH_UCX_MAKE_SEND_TAG(__tag, __rank) \
    ((((uint64_t) (__tag )) << (TORCH_UCX_RANK_BITS)) | \
     (((uint64_t) (__rank))))

#define TORCH_UCX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _src) \
    { \
        if ((_src) == TORCH_UCX_ANY_SOURCE) { \
            _ucp_tag_mask = TORCH_UCX_ANY_SOURCE_MASK; \
        } else { \
            _ucp_tag_mask = TORCH_UCX_SPECIFIC_SOURCE_MASK; \
        } \
        \
        _ucp_tag = ((uint64_t)(_src) & UCS_MASK(TORCH_UCX_RANK_BITS)) | \
                   ((uint64_t)(_tag)) << (TORCH_UCX_RANK_BITS); \
        \
    }


ucs_status_ptr_t torch_ucp_isend(ucp_ep_h ep, void *data, size_t size, int src_rank,
                                 int dst_rank, uint64_t tag);

ucs_status_ptr_t torch_ucp_irecv(ucp_worker_h worker, void *data, size_t size,
                                 int src_rank, uint64_t tag);
