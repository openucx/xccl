/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_H_
#define MCCL_H_
#include <stdint.h>
#include "api/tccl.h"
typedef enum {
    MCCL_SUCCESS,
    MCCL_IN_PROGRESS,
    MCCL_ERROR,
    MCCL_ERR_NOT_SUPPORTED,
} mccl_status_t;

typedef struct mccl_config_t {
    int flags;
    int world_size;
    int world_rank;
    int (*allgather)(void *sbuf, void *rbuf, size_t msglen, void *oob_coll_ctx);
    void* oob_coll_ctx;
} mccl_config_t;

typedef void* mccl_context_h;
typedef void* mccl_comm_h;
typedef void* mccl_request_h;


typedef struct mccl_comm_config_t {
    int (*allgather)(void *sbuf, void *rbuf, size_t msglen, int myrank,
                      int *ranks, int nranks, void *oob_coll_ctx);
    void *oob_coll_ctx;
    mccl_context_h mccl_ctx;
    int is_world;
    int world_rank;
    int comm_size;
    int comm_rank;
    struct {
        uint64_t tagged_colls :1;
    } caps;
} mccl_comm_config_t;

int mccl_init_context(mccl_config_t *conf, mccl_context_h *context);
int mccl_finalize(mccl_context_h context);
int mccl_comm_create(mccl_comm_config_t *conf, mccl_comm_h *comm);
int mccl_comm_free(mccl_comm_h comm);

int mccl_allreduce(void *sbuf, void*rbuf, int count, tccl_dt_t dtype, tccl_op_t op, mccl_comm_h comm);
int mccl_allreduce_init(void *sbuf, void*rbuf, int count, tccl_dt_t dtype, tccl_op_t op, mccl_comm_h comm, mccl_request_h *req);
int mccl_bcast_init(void *buf, int count, tccl_dt_t dtype, int root, mccl_comm_h comm, mccl_request_h *req);
int mccl_bcast(void *buf, int count, tccl_dt_t dtype, int root, mccl_comm_h comm);
int mccl_barrier_init(mccl_comm_h comm, mccl_request_h *req);

/* int mccl_allreduce_tagged_init(void *sbuf, void*rbuf, int count, mccl_datatype_t dtype, mccl_op_t op, mccl_comm_h comm, uint32_t ctag, mccl_request_h *req); */

int mccl_start(mccl_request_h req);
int mccl_test(mccl_request_h req);
int mccl_wait(mccl_request_h req);
int mccl_request_free(mccl_request_h req);
int mccl_progress(mccl_context_h mccl_ctx);
#endif 
