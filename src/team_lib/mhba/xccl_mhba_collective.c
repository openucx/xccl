/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <xccl_mhba_collective.h>
#include "utils/utils.h"

#define TMP_TRANSPOSE_PREALLOC 256

xccl_status_t
xccl_mhba_collective_init_base(xccl_coll_op_args_t *coll_args,
                               xccl_mhba_coll_req_t **request,
                               xccl_mhba_team_t *team)
{
    xccl_mhba_team_t *mhba_team       = ucs_derived_of(team, xccl_mhba_team_t);
    *request = (xccl_mhba_coll_req_t*)malloc(sizeof(xccl_mhba_coll_req_t));
    if (*request == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    memcpy(&((*request)->args), coll_args, sizeof(xccl_coll_op_args_t));
    (*request)->team      = mhba_team;
    (*request)->super.lib = &xccl_team_lib_mhba.super;
    return XCCL_OK;
}

static xccl_status_t
xccl_mhba_node_fanin(xccl_mhba_team_t *team, xccl_mhba_coll_req_t *request)
{
    int i;
    int *ctrl_v;
    int index = SEQ_INDEX(request->seq_num);
    if(team->op_busy[index] && !request->started){
        return XCCL_INPROGRESS;
    } //wait for slot to be open
    team->op_busy[index] = 1;
    request->started = 1;
    xccl_mhba_update_mkeys_entries(&team->node, request); // no option for failure status

    if (team->node.sbgp->group_rank != team->node.asr_rank) {
        *team->node.ops[index].my_ctrl = request->seq_num;
    } else {
        for (i=0; i<team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = (int*)((ptrdiff_t)team->node.ops[index].ctrl + MHBA_CTRL_SIZE*i);
            if (*ctrl_v != request->seq_num) {
                return XCCL_INPROGRESS;
            }
        }
    }
    return XCCL_OK;
}

/* Each rank registers sbuf and rbuf and place the registration data
   in the shared memory location. Next, all rank in node nitify the
   ASR the registration data is ready using SHM Fanin */
static xccl_status_t xccl_mhba_reg_fanin_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    int sr_mem_access_flags = 0;
    int dr_mem_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    xccl_mhba_debug("register memory buffers");

    request->send_bf_mr = ibv_reg_mr(team->node.shared_pd, (void*)request->args.buffer_info.src_buffer,
                                     request->args.buffer_info.len*team->size, sr_mem_access_flags);
    if (!request->send_bf_mr) {
        xccl_mhba_error("Failed to register send_bf memory (errno=%d)", errno);
        return XCCL_ERR_NO_RESOURCE;
    }
    request->receive_bf_mr = ibv_reg_mr(team->node.shared_pd, (void*)request->args.buffer_info.dst_buffer,
                                        request->args.buffer_info.len*team->size, dr_mem_access_flags);
    if (!request->receive_bf_mr) {
        xccl_mhba_error("Failed to register receive_bf memory (errno=%d)", errno);
        ibv_dereg_mr(request->send_bf_mr);
        return XCCL_ERR_NO_RESOURCE;
    }

    xccl_mhba_debug("fanin start");
    /* start task if completion event received */
    task->state = XCCL_TASK_STATE_INPROGRESS;
    /* Start fanin */
    if (XCCL_OK == xccl_mhba_node_fanin(team, request)) {
        xccl_mhba_debug("fanin complete");
        task->state = XCCL_TASK_STATE_COMPLETED;
        xccl_event_manager_notify(&task->em, XCCL_EVENT_COMPLETED);
    } else {
        xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    }
    return XCCL_OK;
}

xccl_status_t xccl_mhba_reg_fanin_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    assert(team->node.sbgp->group_rank == team->node.asr_rank);
    if (XCCL_OK == xccl_mhba_node_fanin(team, request)) {
        xccl_mhba_debug("fanin complete");
        task->state = XCCL_TASK_STATE_COMPLETED;
    }
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_node_fanout(xccl_mhba_team_t *team, xccl_mhba_coll_req_t *request)
{
    int *ctrl_v;
    int index = SEQ_INDEX(request->seq_num);

    /* First phase of fanout: asr signals it completed local ops
       and other ranks wait for asr */
    if (team->node.sbgp->group_rank == team->node.asr_rank) {
        *team->node.ops[index].my_ctrl = request->seq_num;
    } else {
        ctrl_v = (int*)((ptrdiff_t)team->node.ops[index].ctrl + MHBA_CTRL_SIZE*team->node.asr_rank);
        if (*ctrl_v != request->seq_num) {
            return XCCL_INPROGRESS;
        }
    }

    /*Second phase of fanout: wait for remote atomic counters -
      ie wait for the remote data */
    ctrl_v = (int*)((ptrdiff_t)team->node.storage + MHBA_CTRL_SIZE*index);
    assert(*ctrl_v <= team->net.net_size);
    if ( *ctrl_v != team->net.net_size) {
        return XCCL_INPROGRESS;
    }
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_fanout_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    xccl_mhba_debug("fanout start");
    /* start task if completion event received */
    task->state = XCCL_TASK_STATE_INPROGRESS;

    /* Start fanin */
    if (XCCL_OK == xccl_mhba_node_fanout(team, request)) {
        task->state = XCCL_TASK_STATE_COMPLETED;
        xccl_mhba_debug("Algorithm completion");
        team->op_busy[SEQ_INDEX(request->seq_num)] = 0; //todo MT
        xccl_event_manager_notify(&task->em, XCCL_EVENT_COMPLETED);
    } else {
        xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    }
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_fanout_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    if (XCCL_OK == xccl_mhba_node_fanout(team, request)) {
        task->state = XCCL_TASK_STATE_COMPLETED;
        /*Cleanup alg resources - all done */
        xccl_mhba_debug("Algorithm completion");
        team->op_busy[SEQ_INDEX(request->seq_num)] = 0;
    }
    return XCCL_OK;
}

static xccl_status_t xccl_mhba_asr_barrier_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    xccl_mhba_debug("asr barrier start");

    // despite while statement, non blocking because have independent cq, will be finished in a finite time
    xccl_mhba_populate_send_recv_mkeys(team,request);

    //Reset atomic notification counter to 0
    memset(team->node.storage+MHBA_CTRL_SIZE*SEQ_INDEX(request->seq_num),0,MHBA_CTRL_SIZE);

    task->state = XCCL_TASK_STATE_INPROGRESS;
    xccl_coll_op_args_t coll = {
        .coll_type = XCCL_BARRIER,
        .alg.set_by_user = 0,
    };
    //todo create special barrier to support multiple parallel ops - with seq_id
    team->net.ucx_team->ctx->lib->collective_init(&coll, &request->barrier_req,
                                                  team->net.ucx_team);
    team->net.ucx_team->ctx->lib->collective_post(request->barrier_req);
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    return XCCL_OK;
}

xccl_status_t xccl_mhba_asr_barrier_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;

    if (XCCL_OK == team->net.ucx_team->ctx->lib->collective_test(request->barrier_req)) {
        team->net.ucx_team->ctx->lib->collective_finalize(request->barrier_req);
        task->state = XCCL_TASK_STATE_COMPLETED;
    }
    return XCCL_OK;
}

static inline xccl_status_t
send_block_data(struct ibv_qp *qp, uint64_t src_addr, uint32_t msg_size, uint32_t lkey,
                uint64_t remote_addr, uint32_t rkey, int send_flags, int with_imm)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_sge list = {
        .addr	= src_addr,
        .length = msg_size,
        .lkey	= lkey,
    };

    struct ibv_send_wr wr = {
        .wr_id	    = 1,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = with_imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE,
        .send_flags = send_flags,
        .wr.rdma.remote_addr = remote_addr,
        .wr.rdma.rkey =  rkey,
    };

    if (ibv_post_send(qp, &wr, &bad_wr)) {
        xccl_mhba_error("failed to post send");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

static inline xccl_status_t
send_atomic(struct ibv_qp *qp, uint64_t remote_addr, uint32_t rkey,
            xccl_mhba_team_t *team, xccl_mhba_coll_req_t *request)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_sge list = {
            .addr	= (uint64_t) team->dummy_bf_mr->addr,
            .length = team->dummy_bf_mr->length,
            .lkey	= team->dummy_bf_mr->lkey,
    };

    struct ibv_send_wr wr = {
            .wr_id	    = request->seq_num,
            .sg_list    = &list,
            .num_sge    = 1,
            .opcode     = IBV_WR_ATOMIC_FETCH_AND_ADD,
            .send_flags = IBV_SEND_SIGNALED,
            .wr.atomic.remote_addr = remote_addr,
            .wr.atomic.rkey = rkey,
            .wr.atomic.compare_add = 1ULL,
    };

    if (ibv_post_send(qp, &wr, &bad_wr)) {
        xccl_mhba_error("failed to post atomic send");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

static inline void
transpose_square_mat(void* addr, int side_len, int unit_size, void* temp_buffer)
{
    int i,j;
    char tmp_preallocated[TMP_TRANSPOSE_PREALLOC];
    void *tmp = unit_size <= TMP_TRANSPOSE_PREALLOC ? tmp_preallocated : temp_buffer;
    for(i=0;i<side_len-1;i++){
        for(j=i+1;j<side_len;j++){
            memcpy(tmp, addr+(i*unit_size*side_len)+(j*unit_size), unit_size);
            memcpy(addr+(i*unit_size*side_len)+(j*unit_size),
                   addr+(j*unit_size*side_len)+(i*unit_size), unit_size);
            memcpy(addr+j*unit_size*side_len+i*unit_size, tmp, unit_size);
        }
    }
}

static inline xccl_status_t prepost_dummy_recv(struct ibv_qp *qp, int num)
{
    struct ibv_recv_wr wr;
    struct ibv_recv_wr *bad_wr;
    int i;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id      = 0;
    wr.num_sge    = 0;
    for (i=0; i<num; i++) {
        if (ibv_post_recv(qp, &wr, &bad_wr)) {
            xccl_mhba_error("failed to prepost %d receives", num);
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    return XCCL_OK;
}

// add polling mechanism for blocks in order to maintain const qp tx rx
static xccl_status_t xccl_mhba_send_blocks_start_with_transpose(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    size_t len = request->args.buffer_info.len;
    int index = SEQ_INDEX(request->seq_num);
    int node_size = team->node.sbgp->group_size;
    int net_size = team->net.sbgp->group_size;
    int op_msgsize = node_size*team->max_msg_size*team->size;
    int node_msgsize = SQUARED(node_size)*len;
    int block_size = request->block_size;
    int col_msgsize = len*block_size*node_size;
    int block_msgsize = SQUARED(block_size)*len;
    int i, j, k, dest_rank, rank, n_compl, ret;
    uint64_t src_addr, remote_addr;
    struct ibv_wc transpose_completion[1];
    xccl_status_t status;

    xccl_mhba_debug("send blocks start");
    task->state = XCCL_TASK_STATE_INPROGRESS;
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for(i=0;i<net_size;i++) {
        dest_rank = team->net.rank_map[i];
        //send all blocks from curr node to some ARR
        for(j=0;j<xccl_round_up(node_size,block_size);j++){
            for(k=0;k<xccl_round_up(node_size,block_size);k++){
                src_addr = (uintptr_t)(op_msgsize*index + node_msgsize*dest_rank +
                                       col_msgsize*j + block_msgsize*k);
                remote_addr = (uintptr_t)(op_msgsize*index + node_msgsize*rank +
                                          block_msgsize*j + col_msgsize*k);
                prepost_dummy_recv(team->node.umr_qp, 1);
                // SW Transpose
                status = send_block_data(team->node.umr_qp,src_addr,block_msgsize,
                                         team->node.team_send_mkey->lkey,
                                         (uintptr_t)request->transpose_buf_mr->addr,
                                         request->transpose_buf_mr->rkey, IBV_SEND_SIGNALED, 1);
                if(status!=XCCL_OK){
                    xccl_mhba_error("Failed sending block to transpose buffer[%d,%d,%d]",i,j,k);
                    return status;
                }
                n_compl = 0;
                while(n_compl != 2){
                    ret = ibv_poll_cq(team->node.umr_cq, 1, transpose_completion);
                    if (ret > 0) {
                        if (transpose_completion[0].status != IBV_WC_SUCCESS) {
                            xccl_mhba_error("local copy for transpose CQ returned completion with status %s (%d)",
                                            ibv_wc_status_str(transpose_completion[0].status), transpose_completion[0].status);
                            return XCCL_ERR_NO_MESSAGE;
                        }
                        n_compl++;
                    }
                }
                transpose_square_mat(request->transpose_buf_mr->addr,
                                     block_size,request->args.buffer_info.len,
                                     request->tmp_transpose_buf);
                status = send_block_data(team->net.qps[i], (uintptr_t)request->transpose_buf_mr->addr,
                                         block_msgsize, request->transpose_buf_mr->lkey,
                                         remote_addr, team->net.rkeys[i], IBV_SEND_SIGNALED, 0);
                if(status!=XCCL_OK){
                    xccl_mhba_error("Failed sending block [%d,%d,%d]",i,j,k);
                    return status;
                }
                while(!ibv_poll_cq(team->net.cq, 1, transpose_completion)){}
            }
        }
    }

    for(i=0;i<net_size;i++) {
        status = send_atomic(team->net.qps[i],(uintptr_t)team->net.remote_ctrl[i].addr+(index*MHBA_CTRL_SIZE),
                             team->net.remote_ctrl[i].rkey, team, request);
        if(status!=XCCL_OK){
            xccl_mhba_error("Failed sending atomic to node [%d]",i);
            return status;
        }
    }
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    return XCCL_OK;
}

// add polling mechanism for blocks in order to maintain const qp tx rx
static xccl_status_t xccl_mhba_send_blocks_start(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    size_t len = request->args.buffer_info.len;
    int index = SEQ_INDEX(request->seq_num);
    int node_size = team->node.sbgp->group_size;
    int net_size = team->net.sbgp->group_size;
    int op_msgsize = node_size*team->max_msg_size*team->size;
    int node_msgsize = SQUARED(node_size)*len;
    int block_size = request->block_size;
    int col_msgsize = len*block_size*node_size;
    int block_msgsize = SQUARED(block_size)*len;
    int i, j, k, dest_rank, rank;
    uint64_t src_addr, remote_addr;
    xccl_status_t status;

    xccl_mhba_debug("send blocks start");
    task->state = XCCL_TASK_STATE_INPROGRESS;
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for(i=0;i<net_size;i++) {
        dest_rank = team->net.rank_map[i];
        //send all blocks from curr node to some ARR
        for(j=0;j<xccl_round_up(node_size,block_size);j++){
            for(k=0;k<xccl_round_up(node_size,block_size);k++){
                src_addr = (uintptr_t)(op_msgsize*index + node_msgsize*dest_rank + col_msgsize*j + block_msgsize*k);
                remote_addr = (uintptr_t)(op_msgsize*index + node_msgsize*rank +
                                          block_msgsize*j + col_msgsize*k);

                status = send_block_data(team->net.qps[i],src_addr,block_msgsize,team->node.team_send_mkey->lkey,
                                         remote_addr, team->net.rkeys[i],0,0);
                if(status!=XCCL_OK){
                    xccl_mhba_error("Failed sending block [%d,%d,%d]",i,j,k);
                    return status;
                }
            }
        }
        status = send_atomic(team->net.qps[i],(uintptr_t)team->net.remote_ctrl[i].addr+(index*MHBA_CTRL_SIZE),
                             team->net.remote_ctrl[i].rkey, team, request);
        if(status!=XCCL_OK){
            xccl_mhba_error("Failed sending atomic to node [%d]",i);
            return status;
        }
    }
    xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    return XCCL_OK;
}

xccl_status_t xccl_mhba_send_blocks_progress(xccl_coll_task_t *task) {
    xccl_mhba_task_t *self = ucs_derived_of(task, xccl_mhba_task_t);
    xccl_mhba_coll_req_t *request = self->req;
    xccl_mhba_team_t *team = request->team;
    int i, completions_num = ibv_poll_cq(team->net.cq,team->net.sbgp->group_size, team->work_completion);
    if (completions_num<0){
        xccl_mhba_error("ibv_poll_cq() failed for RDMA_ATOMIC execution");
        return XCCL_ERR_NO_MESSAGE;
    }
    for(i=0;i<completions_num;i++){
        if (team->work_completion[i].status != IBV_WC_SUCCESS) {
            xccl_mhba_error("atomic CQ returned completion with status %s (%d)",
                            ibv_wc_status_str(team->work_completion[i].status), team->work_completion[i].status);
            return XCCL_ERR_NO_MESSAGE;
        }
        team->cq_completions[SEQ_INDEX(team->work_completion[i].wr_id)] += 1;
    }
    if(team->cq_completions[SEQ_INDEX(request->seq_num)] == team->net.sbgp->group_size){
        task->state = XCCL_TASK_STATE_COMPLETED;
        team->cq_completions[SEQ_INDEX(request->seq_num)] = 0;
    }
    return XCCL_OK;
}

xccl_status_t
xccl_mhba_alltoall_init(xccl_coll_op_args_t *coll_args,
                        xccl_mhba_coll_req_t *request,
                        xccl_mhba_team_t *team)
{
    xccl_mhba_context_t *ctx = ucs_derived_of(team->super.ctx, xccl_mhba_context_t);
    int is_asr = (team->node.sbgp->group_rank == team->node.asr_rank);
    int n_tasks = (!is_asr) ? 2 : 4;
    size_t len = coll_args->buffer_info.len;
    void *transpose_buf = NULL;
    int i, block_msgsize, block_size;
    xccl_status_t status;
    request->started = 0;
    if (len > team->max_msg_size){
        xccl_mhba_error("msg size too long");
        return XCCL_ERR_NO_RESOURCE;
    }
    xccl_schedule_init(&request->schedule, team->super.ctx);
    if (team->transpose_hw_limitations) {
        block_size = team->blocks_sizes[__ucs_ilog2_u32(len - 1)];
    } else{
        block_size = team->node.sbgp->group_size;
    }

    //todo following section correct assuming homogenous PPN across all nodes
    if(team->node.sbgp->group_size % block_size != 0) {
        if(team->node.sbgp->group_rank == team->node.asr_rank) {
            xccl_mhba_warn("Block size was decreased to fit node PPN");
        }
        while (team->node.sbgp->group_size % block_size && block_size > 2) {
            block_size -= 1;
        }
    }
    block_msgsize = SQUARED(block_size)*len;
    if(team->node.sbgp->group_rank == team->node.asr_rank) {
        xccl_mhba_info("Block size is %d", block_size);
    }
    if(team->node.sbgp->group_size % block_size){
        if(team->node.sbgp->group_rank == team->node.asr_rank) {
            xccl_mhba_error("node PPN can't be divided by any block size, or block size is 1 - NOT SUPPORTED");
        }
        return XCCL_ERR_NO_RESOURCE;
    }
    request->block_size = block_size;
    request->transpose_buf_mr = NULL;
    request->tmp_transpose_buf = NULL;
    request->tasks = (xccl_mhba_task_t*)malloc(sizeof(xccl_mhba_task_t)*n_tasks);
    request->seq_num = team->sequence_number;
    xccl_mhba_debug("Seq num is %d",request->seq_num);
    team->sequence_number++;

    for (i = 0; i < n_tasks; i++) {
        request->tasks[i].req = request;
        xccl_coll_task_init(&request->tasks[i].super);
        if (i > 0) {
            xccl_event_manager_subscribe(&request->tasks[i - 1].super.em, XCCL_EVENT_COMPLETED,
                                         &request->tasks[i].super);
        } else {
            //i == 0
            request->tasks[i].super.handlers[XCCL_EVENT_SCHEDULE_STARTED] = xccl_mhba_reg_fanin_start;
            request->tasks[i].super.progress = xccl_mhba_reg_fanin_progress;
            request->tasks[i].super.handlers[XCCL_EVENT_COMPLETED] = NULL;

            xccl_event_manager_subscribe(&request->schedule.super.em, XCCL_EVENT_SCHEDULE_STARTED,
                                         &request->tasks[i].super);
        }
        xccl_schedule_add_task(&request->schedule, &request->tasks[i].super);
    }
    if (!is_asr) {
        request->tasks[1].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_fanout_start;
        request->tasks[1].super.progress = xccl_mhba_fanout_progress;
    } else {
        request->tasks[1].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_asr_barrier_start;
        request->tasks[1].super.progress = xccl_mhba_asr_barrier_progress;
        if(team->transpose){
            request->tasks[2].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_send_blocks_start_with_transpose;
        } else{
            request->tasks[2].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_send_blocks_start;
        }
        request->tasks[2].super.progress = xccl_mhba_send_blocks_progress;

        request->tasks[3].super.handlers[XCCL_EVENT_COMPLETED] = xccl_mhba_fanout_start;
        request->tasks[3].super.progress = xccl_mhba_fanout_progress;

        if (team->transpose) {
            if (ctx->cfg.transpose_buf_size >= block_msgsize) {
                request->transpose_buf_mr = team->transpose_buf_mr;
            } else {
                transpose_buf = malloc(block_msgsize);
                if (!transpose_buf) {
                    xccl_mhba_error("failed to allocate transpose buffer of %d bytes",
                                    block_msgsize);
                    status = XCCL_ERR_NO_MEMORY;
                    goto tr_buf_error;
                }
                request->transpose_buf_mr =
                    ibv_reg_mr(team->node.shared_pd,
                               transpose_buf, block_msgsize, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                if (!request->transpose_buf_mr) {
                    xccl_mhba_error("failed to register transpose buffer, errno %d", errno);
                    status = XCCL_ERR_NO_MESSAGE;
                    goto tr_buf_reg;
                }
            }
            request->tmp_transpose_buf = NULL;
            if (len > TMP_TRANSPOSE_PREALLOC) {
                request->tmp_transpose_buf = malloc(len);
                if (!request->tmp_transpose_buf) {
                    xccl_mhba_error("failed to allocate tmp transpose buffer of %d bytes",
                                    len);
                    status = XCCL_ERR_NO_MEMORY;
                    goto tmp_buf_error;
                }
            }
        }
    }
    return XCCL_OK;

tmp_buf_error:
    if (transpose_buf) ibv_dereg_mr(request->transpose_buf_mr);
tr_buf_reg:
    free(transpose_buf);
tr_buf_error:
    return status;
}
