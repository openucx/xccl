/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_INLINE_H
#define MCCL_INLINE_H
#include "mccl_core.h"

static inline int sbgp_rank2mccl(sbgp_t *sbgp, int rank) {
    return sbgp->mccl_rank_map[rank];
}

static inline int mccl_comm_rank2world(mccl_comm_t *comm, int rank) {
    if (comm->config.is_world) {
        return rank;
    } else {
        return comm->world_ranks[rank];
    }
}

static inline int sbgp_rank2world(sbgp_t *sbgp, int rank) {
    return mccl_comm_rank2world(sbgp->mccl_comm, sbgp_rank2mccl(sbgp, rank));
}

static inline int is_rank_on_local_node(int rank, mccl_comm_t *mccl_comm) {
    mccl_context_t *ctx = mccl_comm->config.mccl_ctx;
    return ctx->procs[mccl_comm_rank2world(mccl_comm, rank)].node_hash == ctx->local_proc.node_hash;
}

static inline int is_rank_on_local_socket(int rank, mccl_comm_t *mccl_comm) {
    mccl_context_t *ctx = mccl_comm->config.mccl_ctx;
    if (ctx->local_proc.socketid < 0) {
        return 0;
    }
    proc_data_t *proc = &ctx->procs[mccl_comm_rank2world(mccl_comm, rank)];
    return proc->node_hash == ctx->local_proc.node_hash &&
        proc->socketid == ctx->local_proc.socketid;
}

#endif
