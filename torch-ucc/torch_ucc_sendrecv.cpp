#include "torch_ucc_sendrecv.hpp"


static void torch_ucp_send_handler(void *request, ucs_status_t status)
{
  return;
}

static void torch_ucp_recv_handler(void *request, ucs_status_t status,
                             ucp_tag_recv_info_t *info)
{
  return;
}

ucs_status_ptr_t torch_ucp_isend(ucp_ep_h ep, void *data, size_t size, int src_rank,
                                 int dst_rank, uint64_t tag)
{
  ucp_tag_t ucp_tag = TORCH_UCX_MAKE_SEND_TAG(tag, src_rank);

  return ucp_tag_send_nb(ep, data, size, ucp_dt_make_contig(1), ucp_tag,
                         torch_ucp_send_handler);
}


ucs_status_ptr_t torch_ucp_irecv(ucp_worker_h worker, void *data, size_t size,
                                 int src_rank, uint64_t tag)
{
  ucp_tag_t tag_mask;
  ucp_tag_t ucp_tag;

  TORCH_UCX_MAKE_RECV_TAG(ucp_tag, tag_mask, tag, src_rank);

  return ucp_tag_recv_nb(worker, data, size, ucp_dt_make_contig(1), ucp_tag,
                         tag_mask, torch_ucp_recv_handler);
}
