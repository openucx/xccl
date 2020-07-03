#pragma once

#include <torch/extension.h>

#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <pybind11/chrono.h>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

#include <ucp/api/ucp.h>
#include <api/xccl.h>

namespace c10d {

class ProcessGroupUCC : public ProcessGroup {
 public:
  class WorkUCP : public ProcessGroup::Work {
   public:
    WorkUCP(ucs_status_ptr_t request, ucp_worker_h ucp_worker):
            req(request), worker(ucp_worker) {}

    virtual ~WorkUCP();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait() override;

   protected:
    ucs_status_ptr_t req;
    ucp_worker_h worker;
    friend class ProcessGroupUCC;
  };

  class WorkUCC : public ProcessGroup::Work {
   public:
    WorkUCC(xccl_coll_req_h request, uint32_t *sl, uint32_t *rl, uint32_t *so, uint32_t *ro)
                                     :req(request),
                                     send_lengths(sl), recv_lengths(rl),
                                     send_offsets(so), recv_offsets(ro) {}
 
    WorkUCC(xccl_coll_req_h request):req(request)
    {
      send_lengths = recv_lengths = NULL;
      send_offsets = recv_offsets = NULL;
    }

    virtual ~WorkUCC();
    bool isCompleted() override;
    bool isSuccess() const override;
    bool wait() override;

   protected:
    xccl_coll_req_h req;
    uint32_t     *send_lengths;
    uint32_t     *recv_lengths;
    uint32_t     *send_offsets;
    uint32_t     *recv_offsets;
    friend class ProcessGroupUCC;
  };


  explicit ProcessGroupUCC(const std::shared_ptr<Store>& store,
                           int rank = -1,
                           int size = -1);
  virtual ~ProcessGroupUCC();

  std::shared_ptr<ProcessGroup::Work> broadcast(std::vector<at::Tensor>& data,
                                                const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(std::vector<at::Tensor>& tensors,
                                                const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(std::vector<at::Tensor>& tensors,
                                                          const AllreduceCoalescedOptions& opts = AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(std::vector<at::Tensor>& tensors,
                                             const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                std::vector<at::Tensor>& inputTensors,
                                                const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(at::Tensor& outputBuffer,
                                                     at::Tensor& inputBuffer,
                                                     const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> barrier(const BarrierOptions& opts = BarrierOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                             std::vector<at::Tensor>& inputTensors,
                                             const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(std::vector<at::Tensor>& outputTensors,
                                              std::vector<std::vector<at::Tensor>>& inputTensors,
                                              const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(std::vector<at::Tensor>& outputTensors,
                                                     std::vector<std::vector<at::Tensor>>& inputTensors,
                                                     const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall_base(at::Tensor& outputTensor,
                                                    at::Tensor& inputTensor,
                                                    std::vector<int64_t>& outputSplitSizes,
                                                    std::vector<int64_t>& inputSplitSizes,
                                                    const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall(std::vector<at::Tensor>& outputTensors,
                                               std::vector<at::Tensor>& inputTensors,
                                               const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(std::vector<at::Tensor>& tensors,
                                           int dstRank,
                                           int tag);

  std::shared_ptr<ProcessGroup::Work> recv(std::vector<at::Tensor>& tensors,
                                           int srcRank,
                                           int tag);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(std::vector<at::Tensor>& tensors,
                                                    int tag);

  static std::shared_ptr<ProcessGroup> createProcessGroupUCC(
      const std::shared_ptr<::c10d::Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

  static void ProcessGroupUCCConstructor() __attribute__((constructor)) {
      py::object module = py::module::import("torch.distributed");
      py::object register_backend = module.attr("Backend").attr("register_backend");
      register_backend("ucc", py::cpp_function(createProcessGroupUCC));
  }

protected:
  std::shared_ptr<Store> store_;

  ucp_context_h ucp_ctx;
  ucp_worker_h ucp_worker;
  std::vector<ucp_ep_h> ucp_eps;

  xccl_lib_h xccl_lib;
  xccl_context_h xccl_ctx;
  xccl_team_h xccl_team;
private:
  struct xccl_oob_allgather_req_t {
      xccl_ep_range_t range;
      void *sbuf;
      void *rbuf;
      void *oob_coll_ctx;
      int my_rank;
      size_t msglen;
      int iter;
      ucs_status_ptr_t reqs[2];
  };

  struct xccl_oob_coll_ctx_t {
      int          rank;
      int          size;
      ucp_worker_h ucp_worker;
      ucp_ep_h     *ucp_eps;
  } oob_coll_ctx;

  void check_tensor(const std::vector<at::Tensor>& tensors);
  xccl_coll_req_h launch_xccl_collective(xccl_collective_type_t coll,
                                         const std::vector<at::Tensor>& tensors,
                                         int root, xccl_op_t op);
  static ucs_status_t ucp_test_all(ucp_worker_h worker, int n_reqs,
                                   ucs_status_ptr_t *reqs, int *completed);
  static xccl_status_t oob_allgather_test(void *req);
  static xccl_status_t oob_allgather_free(void *req);
  static int oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                     int my_rank, xccl_ep_range_t range,
                                     void *oob_coll_ctx, void **req);
  void init_xccl();
};

} // namespace c10d
