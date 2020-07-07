#include "torch_ucc.hpp"
#include "torch_ucc_sendrecv.hpp"
#include <map>
#include <iostream>
#include <stdio.h>

namespace c10d {

std::map<ReduceOp, xccl_op_t> xccl_op_map = {
    {ReduceOp::MIN,     XCCL_OP_MIN},
    {ReduceOp::MAX,     XCCL_OP_MAX},
    {ReduceOp::SUM,     XCCL_OP_SUM},
    {ReduceOp::PRODUCT, XCCL_OP_PROD},
};

std::map<at::ScalarType, xccl_dt_t> xccl_type_map = {
    {at::kByte,   XCCL_DT_UINT8},
    {at::kChar,   XCCL_DT_INT8},
    {at::kHalf,   XCCL_DT_FLOAT16},
    {at::kDouble, XCCL_DT_FLOAT64},
    {at::kFloat,  XCCL_DT_FLOAT32},
    {at::kInt,    XCCL_DT_INT32},
    {at::kLong,   XCCL_DT_INT64},
};

void ProcessGroupUCC::check_tensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error("ProcessGroupUCC takes 1 tensoe");
  }
  if (!tensors[0].is_contiguous()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be contiguous");
  }
  if (tensors[0].is_sparse()) {
    throw std::runtime_error("ProcessGroupUCC input tensor has to be dense");
  }
  //TODO: check cuda case
}

xccl_coll_req_h ProcessGroupUCC::launch_xccl_collective(xccl_collective_type_t coll,
                                                       const std::vector<at::Tensor>& tensors,
                                                       int root, xccl_op_t op)
{
  xccl_coll_op_args_t coll_args;
  xccl_coll_req_h     request;

  auto &tensor = tensors[0];
  coll_args.coll_type              = coll;
  coll_args.buffer_info.src_buffer = tensor.data_ptr();
  coll_args.buffer_info.dst_buffer = tensor.data_ptr();
  coll_args.buffer_info.len        = tensor.numel() * tensor.element_size();

  if ((coll == XCCL_BCAST) || (coll == XCCL_REDUCE)) {
      coll_args.root               = root;
  }

  if ((coll == XCCL_REDUCE) || (coll == XCCL_ALLREDUCE)) {
    coll_args.reduce_info.dt       = xccl_type_map.at(tensor.scalar_type());
    coll_args.reduce_info.op       = op;
    coll_args.reduce_info.count    = tensor.numel();
  }

  coll_args.alg.set_by_user        = 0;
  coll_args.tag                    = 123;

  xccl_collective_init(&coll_args, &request, xccl_team);
  xccl_collective_post(request);
  return request;
}

ProcessGroupUCC::WorkUCP::~WorkUCP()
{
  if (req != NULL) {
    ucp_request_free(req);
  }
}

bool ProcessGroupUCC::WorkUCP::isCompleted()
{
  return (req == NULL) ? true: (ucp_request_check_status(req) != UCS_INPROGRESS);
}

bool ProcessGroupUCC::WorkUCP::isSuccess() const
{
  //TODO
  return true;
}

bool ProcessGroupUCC::WorkUCP::wait() {
  do {
    ucp_worker_progress(worker);
  } while (!isCompleted());
  return true;
}

ProcessGroupUCC::WorkUCC::~WorkUCC()
{
  xccl_collective_finalize(req);
}

bool ProcessGroupUCC::WorkUCC::isCompleted()
{
  xccl_status_t st;

  st = xccl_collective_test(req);
  
  return st != XCCL_INPROGRESS;
}

bool ProcessGroupUCC::WorkUCC::isSuccess() const
{
  return true;
}

bool ProcessGroupUCC::WorkUCC::wait()
{
  xccl_status_t st;

  st = xccl_collective_wait(req);

  if (args.coll_type == XCCL_ALLGATHER) {
    for (size_t i = 0; i < output_data_vec.size(); ++i) {
      (output_data_vec)[i].copy_(flat_tensor[i]);
    }

  }

  return st == XCCL_OK;
}

ucs_status_t ProcessGroupUCC::ucp_test_all(ucp_worker_h worker, int n_reqs,
                                           ucs_status_ptr_t *reqs,
                                           int *completed)
{
  int n_completed = 0;

  for (int i = 0; i < n_reqs; i++) {
    if (reqs[i] == NULL) {
      n_completed++;
    }
    else {
      if (ucp_request_check_status(reqs[i]) != UCS_INPROGRESS) {
        ucp_request_free(reqs[i]);
        reqs[i] = NULL;
        n_completed++;
      }
      else {
        ucp_worker_progress(worker);
      }
    }
  }

  *completed =  (n_completed == n_reqs) ? 1 : 0;

  return UCS_OK;
}

xccl_status_t ProcessGroupUCC::oob_allgather_test(void *req)
{
  xccl_oob_allgather_req_t *oob_req = static_cast<xccl_oob_allgather_req_t*>(req);
  int rank, size, sendto, recvfrom, recvdatafrom, senddatafrom, completed, probe;
  char *tmpsend = NULL, *tmprecv = NULL;
  size_t msglen = oob_req->msglen;
  const int probe_count = 1;
  xccl_oob_coll_ctx_t *oob_ctx = static_cast<xccl_oob_coll_ctx_t*>(oob_req->oob_coll_ctx);

  if (oob_req->range.type == XCCL_EP_RANGE_UNDEFINED) {
    size = oob_ctx->size;
    rank = oob_ctx->rank;
  } else {
    size = oob_req->range.ep_num;
    rank = oob_req->my_rank;
  }

  if (oob_req->iter == 0) {
    tmprecv = (char*) oob_req->rbuf + (ptrdiff_t)rank * (ptrdiff_t)msglen;
    memcpy(tmprecv, oob_req->sbuf, msglen);
  }
  sendto = (rank + 1) % size;
  recvfrom  = (rank - 1 + size) % size;
  if (oob_req->range.type != XCCL_EP_RANGE_UNDEFINED) {
    sendto   = xccl_range_to_rank(oob_req->range, sendto);
    recvfrom = xccl_range_to_rank(oob_req->range, recvfrom);
  }
  for (; oob_req->iter < size - 1; oob_req->iter++) {
    if (oob_req->iter > 0) {
      probe = 0;
      do {
        ucp_test_all(oob_ctx->ucp_worker, 2, oob_req->reqs, &completed);
        probe++;
      } while (!completed && probe < probe_count);
      if (!completed) {
        return XCCL_INPROGRESS;
      }
    }
    recvdatafrom = (rank - oob_req->iter - 1 + size) % size;
    senddatafrom = (rank - oob_req->iter + size) % size;
    tmprecv = (char*)oob_req->rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)msglen;
    tmpsend = (char*)oob_req->rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)msglen;

    oob_req->reqs[0] = torch_ucp_isend(oob_ctx->ucp_eps[sendto], tmpsend, msglen,
                                       rank, sendto, TORCH_XCCL_OOB_TAG);
    oob_req->reqs[1] = torch_ucp_irecv(oob_ctx->ucp_worker, tmprecv, msglen,
                                       recvfrom, TORCH_XCCL_OOB_TAG);

  }

  probe = 0;
  do {
    ucp_test_all(oob_ctx->ucp_worker, 2, oob_req->reqs, &completed);
    probe++;
  } while (!completed && probe < probe_count);
  if (!completed) {
    return XCCL_INPROGRESS;
  }

  return XCCL_OK;
}

xccl_status_t ProcessGroupUCC::oob_allgather_free(void *req)
{
  xccl_oob_allgather_req_t *request = static_cast<xccl_oob_allgather_req_t*>(req);
  delete request;

  return XCCL_OK;
}

int ProcessGroupUCC::oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                   int my_rank, xccl_ep_range_t range,
                                   void *oob_coll_ctx, void **req)
{
  xccl_oob_allgather_req_t *oob_req = new(xccl_oob_allgather_req_t);
  oob_req->sbuf         = sbuf;
  oob_req->rbuf         = rbuf;
  oob_req->msglen       = msglen;
  oob_req->range        = range;
  oob_req->oob_coll_ctx = oob_coll_ctx;
  oob_req->my_rank      = my_rank;
  oob_req->iter         = 0;
  *req = oob_req;

  return oob_allgather_test(oob_req);
}

void ProcessGroupUCC::init_xccl()
{
  xccl_lib_params_t lib_params;
  xccl_lib_config_t *cfg;
  xccl_status_t     st;

  memset(&lib_params, 0, sizeof(lib_params));
  lib_params.field_mask = XCCL_LIB_PARAM_FIELD_TEAM_USAGE |
                          XCCL_LIB_PARAM_FIELD_COLL_TYPES;

  lib_params.team_usage = XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES |
                          XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES;

  lib_params.coll_types = XCCL_COLL_CAP_BCAST |
                          XCCL_COLL_CAP_ALLREDUCE |
                          XCCL_COLL_CAP_ALLTOALL |
                          XCCL_COLL_CAP_ALLTOALLV;

  cfg = NULL;
  st = xccl_lib_init(&lib_params, cfg, &xccl_lib);
  if (st != XCCL_OK) {
    throw std::runtime_error("Failed to init xccl lib");
  }

  xccl_context_config_t *ctx_config;
  st = xccl_context_config_read(xccl_lib, "TORCH", NULL, &ctx_config);
  if (st != XCCL_OK) {
    xccl_lib_cleanup(xccl_lib);
    throw std::runtime_error("Failed to read xccl config");
  }
  
  xccl_context_params_t ctx_params;

  oob_coll_ctx.ucp_eps    = ucp_eps.data();
  oob_coll_ctx.ucp_worker = ucp_worker;
  oob_coll_ctx.rank       = rank_;
  oob_coll_ctx.size       = size_;

  ctx_params.field_mask       = XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE |
                                XCCL_CONTEXT_PARAM_FIELD_OOB |
                                XCCL_CONTEXT_PARAM_FIELD_TEAM_COMPLETION_TYPE |
                                XCCL_CONTEXT_PARAM_FIELD_TLS;

  ctx_params.thread_mode      = XCCL_THREAD_MODE_MULTIPLE;

  ctx_params.completion_type  = XCCL_TEAM_COMPLETION_TYPE_BLOCKING;

  ctx_params.tls              = XCCL_TL_UCX;

  ctx_params.oob.allgather    = oob_allgather;
  ctx_params.oob.req_test     = oob_allgather_test;
  ctx_params.oob.req_free     = oob_allgather_free;
  ctx_params.oob.coll_context = static_cast<void*>(&oob_coll_ctx);
  ctx_params.oob.rank         = rank_;
  ctx_params.oob.size         = size_;

  st = xccl_context_create(xccl_lib, &ctx_params, ctx_config, &xccl_ctx);
  xccl_context_config_release(ctx_config);
  if (st != XCCL_OK) {
    xccl_lib_cleanup(xccl_lib);
    throw std::runtime_error("Failed to create xccl context");
  }

  xccl_team_params_t team_params;

  team_params.field_mask           = XCCL_TEAM_PARAM_FIELD_EP_RANGE |
                                     XCCL_TEAM_PARAM_FIELD_OOB;

  team_params.range.type           = XCCL_EP_RANGE_STRIDED;
  team_params.range.strided.start  = 0;
  team_params.range.strided.stride = 1;
  team_params.oob.allgather        = oob_allgather;
  team_params.oob.req_test         = oob_allgather_test;
  team_params.oob.req_free         = oob_allgather_free;
  team_params.oob.coll_context     = static_cast<void*>(&oob_coll_ctx);
  team_params.oob.rank             = rank_;
  team_params.oob.size             = size_;

  st = xccl_team_create_post(xccl_ctx, &team_params, &xccl_team);
  if (st != XCCL_OK) {
    xccl_context_destroy(xccl_ctx);
    xccl_lib_cleanup(xccl_lib);
  }
  while (XCCL_INPROGRESS == xccl_team_create_test(xccl_team)) {};

}
ProcessGroupUCC::ProcessGroupUCC(const std::shared_ptr<Store>& store,
                                 int rank,
                                 int size)
    : ProcessGroup(rank, size),
      store_(store) {
  ucp_params_t params;
  ucp_config_t *config;
  ucs_status_t st;

  st = ucp_config_read("TORCH", NULL, &config);
  if (st != UCS_OK) {
    throw std::runtime_error("Failed to read ucp config");
  }

  memset(&params, 0, sizeof(ucp_params_t));
  params.field_mask        = UCP_PARAM_FIELD_FEATURES |
                             UCP_PARAM_FIELD_REQUEST_SIZE |
                             UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;
  params.features          = UCP_FEATURE_TAG;
  params.request_size      = sizeof(WorkUCP);
  params.estimated_num_eps = size;
  st = ucp_init(&params, config, &ucp_ctx);
  ucp_config_release(config);
  if (st != UCS_OK) {
    throw std::runtime_error("Failed to init ucp context");
  }

  ucp_worker_params_t worker_params;
  memset(&worker_params, 0, sizeof(ucp_worker_params_t));
  worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
  st = ucp_worker_create(ucp_ctx, &worker_params, &ucp_worker);
  if (st != UCS_OK) {
    ucp_cleanup(ucp_ctx);
    throw std::runtime_error("Failed to init ucp worker");
  }
  // TODO: check that multithread support is provided

  ucp_address_t *local_addr;
  size_t local_addr_len;
  st = ucp_worker_get_address(ucp_worker, &local_addr, &local_addr_len);
  if (st != UCS_OK) {
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_ctx);
    throw std::runtime_error("Failed to get ucp worker address");
  }

  auto key = "wa" + std::to_string(rank);
  auto val = std::vector<uint8_t>(reinterpret_cast<uint8_t*>(local_addr),
                                  reinterpret_cast<uint8_t*>(local_addr) + local_addr_len);
  store_->set(key, val);
  ucp_worker_release_address(ucp_worker, local_addr);
  ucp_eps.resize(size);
  
  for(int i = 0; i < size; i++) {
    ucp_ep_params_t ep_params;
    std::vector<uint8_t> peer_addr = store_->get("wa"+std::to_string(i));

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = reinterpret_cast<ucp_address_t*>(peer_addr.data());
    st = ucp_ep_create(ucp_worker, &ep_params, &ucp_eps[i]);
    if (st != UCS_OK) {
        //TODO: add error handling and cleanup
        throw std::runtime_error("Failed to create ep to rank");
    }
  }
  init_xccl();
}

ProcessGroupUCC::~ProcessGroupUCC()
{
  ucs_status_ptr_t close_req;
  ucs_status_t     st;

  xccl_team_destroy(xccl_team);
  xccl_context_destroy(xccl_ctx);
  xccl_lib_cleanup(xccl_lib);

  for (int i = 0; i < size_; i++) {
    close_req = ucp_ep_close_nb(ucp_eps[i], UCP_EP_CLOSE_MODE_FLUSH);
    if (UCS_PTR_IS_ERR(close_req)) {
      return;
    }
    if (UCS_PTR_IS_PTR(close_req)) {
      do {
        ucp_worker_progress(ucp_worker);
        st = ucp_request_check_status(close_req);
      } while (st != UCS_OK);
      ucp_request_free(close_req);
    }
  }
  
  auto key = "close" + std::to_string(rank_);
  auto val = std::vector<uint8_t>{0xFF};
  store_->set(key, val);
  std::vector<std::string> peer_keys(size_);
  for (int i = 0; i < size_; i++) {
    peer_keys[i] = "close" + std::to_string(i);
  }
  try {
    store_->wait(peer_keys, std::chrono::milliseconds(100));
  }
  catch(...) {
  }
  ucp_worker_destroy(ucp_worker);
  ucp_cleanup(ucp_ctx);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::broadcast(std::vector<at::Tensor>& tensors,
                                                               const BroadcastOptions& opts)
{
  xccl_coll_req_h request;
  
  request = launch_xccl_collective(XCCL_BCAST, tensors, opts.rootRank,
                                   XCCL_OP_LAST_PREDEFINED);
  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce(std::vector<at::Tensor>& tensors,
                                                               const AllreduceOptions& opts)
{
  xccl_coll_req_h request;
  
  request = launch_xccl_collective(XCCL_ALLREDUCE, tensors, -1,
                                   xccl_op_map.at(opts.reduceOp));
  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allreduce_coalesced(std::vector<at::Tensor>& tensors,
                                                                         const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce(std::vector<at::Tensor>& tensors,
                                                            const ReduceOptions& opts)
{
  xccl_coll_req_h request;
  
  request = launch_xccl_collective(XCCL_REDUCE, tensors, opts.rootRank,
                                   xccl_op_map.at(opts.reduceOp));
  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                              std::vector<at::Tensor>& inputTensors,
                                                              const AllgatherOptions& opts)
{
  auto req     = std::make_shared<ProcessGroupUCC::WorkUCC>();
  auto &tensor = inputTensors[0];
  xccl_coll_op_args_t coll_args;
  xccl_coll_req_h     request;

  req->flat_tensor = newLikeFlat(outputTensors[0]);
  req->output_data_vec = (outputTensors[0]);
  coll_args.coll_type              = XCCL_ALLGATHER;
  coll_args.buffer_info.src_buffer = tensor.data_ptr();
  coll_args.buffer_info.dst_buffer = req->flat_tensor.data_ptr();
  coll_args.buffer_info.len        = tensor.numel() * tensor.element_size() * size_;
  coll_args.alg.set_by_user        = 0;
  coll_args.tag                    = 123;

  xccl_collective_init(&coll_args, &request, xccl_team);
  xccl_collective_post(request);
  req->args = coll_args;
  req->req = request;

  return req;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::allgather_base(at::Tensor& outputBuffer,
                                                                    at::Tensor& inputBuffer,
                                                                    const AllgatherOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::barrier(
    const BarrierOptions& opts) {
  xccl_coll_req_h request;
  xccl_coll_op_args_t coll_args;

  coll_args.coll_type = XCCL_BARRIER;

  xccl_collective_init(&coll_args, &request, xccl_team);
  xccl_collective_post(request);

  return std::make_shared<ProcessGroupUCC::WorkUCC>(request);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::gather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const GatherOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::scatter(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<std::vector<at::Tensor>>& inputTensors,
                                                             const ScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::reduce_scatter(std::vector<at::Tensor>& outputTensors,
                                                                    std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                    const ReduceScatterOptions& opts) {
  throw std::runtime_error("ProcessGroupUCC does not support reduce_scatter");
}


int64_t computeLengthsAndOffsets(int group_size,
                                 const std::vector<int64_t>& split_sizes,
                                 const at::Tensor& tensor,
                                 uint32_t* lengths,
                                 uint32_t* offsets)
{
  bool equal_splits = false;
  int64_t dim0_size = tensor.size(0);
  int64_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
  int64_t split_size = 0;
  int64_t offset = 0;

  if (split_sizes.size() == 0) {
    equal_splits = true;
    split_size = tensor.size(0) / group_size;
  }

  for (int i = 0; i < group_size; i++) {
    int64_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
    lengths[i] = length;
    offsets[i] = offset;
    offset += length;
  }
  return offset;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall_base(at::Tensor& outputTensor,
                                                                   at::Tensor& inputTensor,
                                                                   std::vector<int64_t>& outputSplitSizes,
                                                                   std::vector<int64_t>& inputSplitSizes,
                                                                   const AllToAllOptions& opts)
{
  auto req = std::make_shared<ProcessGroupUCC::WorkUCC>();
  xccl_coll_req_h     request;
  xccl_coll_op_args_t coll_args;

  if ((outputSplitSizes.size() == 0) || (inputSplitSizes.size() == 0)) {
    coll_args.coll_type              = XCCL_ALLTOALL;
    coll_args.buffer_info.src_buffer = inputTensor.data_ptr();
    coll_args.buffer_info.dst_buffer = outputTensor.data_ptr();
    coll_args.buffer_info.len        = inputTensor.element_size() * inputTensor.numel() / size_;
    coll_args.alg.set_by_user        = 0;
    coll_args.tag                    = 123;
  } else {
    req->scratch.resize(4 * size_);
    uint32_t *send_lengths = req->scratch.data();
    uint32_t *recv_lengths = (uint32_t*)((ptrdiff_t)send_lengths + 1*size_*sizeof(uint32_t));
    uint32_t *send_offsets = (uint32_t*)((ptrdiff_t)send_lengths + 2*size_*sizeof(uint32_t));
    uint32_t *recv_offsets = (uint32_t*)((ptrdiff_t)send_lengths + 3*size_*sizeof(uint32_t));

    computeLengthsAndOffsets(size_, inputSplitSizes, inputTensor, send_lengths, send_offsets);
    computeLengthsAndOffsets(size_, outputSplitSizes, outputTensor, recv_lengths, recv_offsets);

    coll_args.coll_type                     = XCCL_ALLTOALLV;
    coll_args.buffer_info.src_buffer        = inputTensor.data_ptr();
    coll_args.buffer_info.src_displacements = send_offsets;
    coll_args.buffer_info.src_counts        = send_lengths;
    coll_args.buffer_info.src_datatype      = xccl_type_map.at(inputTensor.scalar_type());
    coll_args.buffer_info.dst_buffer        = outputTensor.data_ptr();
    coll_args.buffer_info.dst_displacements = recv_offsets;
    coll_args.buffer_info.dst_counts        = recv_lengths;
    coll_args.buffer_info.dst_datatype      = xccl_type_map.at(outputTensor.scalar_type());
    coll_args.alg.set_by_user               = 0;
    coll_args.tag                           = 123;
  }

  xccl_collective_init(&coll_args, &request, xccl_team);
  xccl_collective_post(request);

  req->args = coll_args;
  req->req  = request;

  return req;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::alltoall(std::vector<at::Tensor>& outputTensors,
                                                              std::vector<at::Tensor>& inputTensors,
                                                              const AllToAllOptions& opts)
{
  throw std::runtime_error("ProcessGroupUCC does not support alltoall");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::send(std::vector<at::Tensor>& tensors,
                                                          int dstRank,
                                                          int tag)
{
  //TODO: check tensor count and type, assume single dense tensor
  auto &tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  ucs_status_ptr_t st;

  st = torch_ucp_isend(ucp_eps[dstRank], tensor.data_ptr(), size, rank_, dstRank, tag);
  if (UCS_PTR_IS_ERR(st)) {
    throw std::runtime_error("Failed to send msg");
  }
  
  return std::make_shared<ProcessGroupUCC::WorkUCP>(st, ucp_worker);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::recv(std::vector<at::Tensor>& tensors,
                                                          int srcRank,
                                                          int tag)
{
  auto &tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  ucs_status_ptr_t st;

  st = torch_ucp_irecv(ucp_worker, tensor.data_ptr(), size, srcRank, tag);
  if (UCS_PTR_IS_ERR(st)) {
    throw std::runtime_error("Failed to recv msg");
  }

  return std::make_shared<ProcessGroupUCC::WorkUCP>(st, ucp_worker);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupUCC::recvAnysource(std::vector<at::Tensor>& tensors,
                                                                   int tag)
{
  auto &tensor = tensors[0];
  size_t size = tensor.numel() * tensor.element_size();
  ucs_status_ptr_t st;

  st = torch_ucp_irecv(ucp_worker, tensor.data_ptr(), size, TORCH_UCX_ANY_SOURCE, tag);
  if (UCS_PTR_IS_ERR(st)) {
    throw std::runtime_error("Failed to recv msg");
  }

  return std::make_shared<ProcessGroupUCC::WorkUCP>(st, ucp_worker);
}

std::shared_ptr<ProcessGroup> ProcessGroupUCC::createProcessGroupUCC(
    const std::shared_ptr<::c10d::Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout) {
  return std::make_shared<ProcessGroupUCC>(store, rank, size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createProcessGroupUCC", &ProcessGroupUCC::createProcessGroupUCC);
}

} // namespace c10d
