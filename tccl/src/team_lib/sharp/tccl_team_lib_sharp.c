/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "tccl_team_lib_sharp.h"
#include "tccl_sharp_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>

unsigned int tccl_sharp_global_rand_state;
static inline int tccl_sharp_rand()
{
    return rand_r(&tccl_sharp_global_rand_state);
}

static inline void tccl_sharp_rand_state_init(unsigned int *state)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    *state = (unsigned int)(tval.tv_usec ^ getpid());
}

static inline void tccl_sharp_global_rand_state_init()
{
    tccl_sharp_rand_state_init(&tccl_sharp_global_rand_state);
}

int tccl_team_sharp_oob_barrier(void *context)
{
    tccl_oob_collectives_t *oob = (tccl_oob_collectives_t*)context;
    int comm_size = oob->size;
    int comm_rank = oob->rank;
    char *tmp = NULL, c = 'c';
    int rc;

    tmp = (char*)malloc(comm_size * sizeof(char));
    rc = oob->allgather(&c, tmp, sizeof(char), oob->coll_context);
    free(tmp);
    return rc;
}

int tccl_team_sharp_oob_gather(void *context, int root, void *sbuf,
                              void *rbuf, int size)
{
    tccl_oob_collectives_t *oob = (tccl_oob_collectives_t*)context;
    int comm_size = oob->size;
    int comm_rank = oob->rank;
    void *tmp = NULL;
    int rc;

    if (comm_rank != root) {
        tmp = malloc(comm_size*size);
        rbuf = tmp;
    }
    rc = oob->allgather(sbuf, rbuf, size, oob->coll_context);
    if (tmp) {
        free(tmp);
    }
    return rc;
}

int tccl_team_sharp_oob_bcast(void *context, void *buf, int size, int root)
{
    tccl_oob_collectives_t *oob = (tccl_oob_collectives_t*)context;
    int comm_size = oob->size;
    int comm_rank = oob->rank;
    void *tmp;
    int rc; 

    tmp = malloc(comm_size*size);
    rc = oob->allgather(buf, tmp, size, oob->coll_context);
    if (comm_rank != root) {
        memcpy(buf, (void*)((ptrdiff_t)tmp + root*size), size);
    }
    free(tmp);
    return rc;
}

static tccl_status_t
tccl_team_sharp_create_context(tccl_team_lib_h team_lib,
                              tccl_team_context_config_h config,
                              tccl_team_context_h *team_context)
{
    tccl_team_sharp_context_t *team_sharp_ctx = malloc(sizeof(*team_sharp_ctx));
    struct sharp_coll_init_spec init_spec = {0};

    tccl_sharp_global_rand_state_init();
    map_tccl_to_sharp_dtype();
    map_tccl_to_sharp_reduce_op_type();
    team_sharp_ctx->oob = config->oob;

    init_spec.progress_func                  = NULL;
    init_spec.world_rank                     = team_sharp_ctx->oob.rank;
    init_spec.world_local_rank               = 0;
    init_spec.world_size                     = team_sharp_ctx->oob.size;
    init_spec.enable_thread_support          = 1;
    init_spec.group_channel_idx              = 0;
    init_spec.oob_colls.barrier              = tccl_team_sharp_oob_barrier;
    init_spec.oob_colls.bcast                = tccl_team_sharp_oob_bcast;
    init_spec.oob_colls.gather               = tccl_team_sharp_oob_gather;
    init_spec.oob_ctx                        = &team_sharp_ctx->oob;
    init_spec.config                         = sharp_coll_default_config;
    init_spec.config.user_progress_num_polls = 1000000;
    init_spec.config.ib_dev_list             = "mlx5_0:1";
    init_spec.job_id                         = tccl_sharp_rand();
    tccl_team_sharp_oob_bcast((void*)&team_sharp_ctx->oob, &init_spec.job_id,
                              sizeof(uint64_t), 0);
    int ret = sharp_coll_init(&init_spec, &team_sharp_ctx->sharp_context);
    if (ret < 0 ) {
        if (team_sharp_ctx->oob.rank == 0) {
            fprintf(stderr, "Failed to initialize SHARP collectives:%s(%d)"  
                            "job ID:%" PRIu64"\n",
                            sharp_coll_strerror(ret), ret, init_spec.job_id);
        }
        free(team_sharp_ctx);
        return TCCL_ERR_NO_MESSAGE;
    }
    *team_context = (tccl_team_context_h)team_sharp_ctx;
    return TCCL_OK;
}

static tccl_status_t
tccl_team_sharp_destroy_context(tccl_team_context_h team_context)
{
    tccl_team_sharp_context_t *team_sharp_ctx =
        (tccl_team_sharp_context_t *)team_context;
    if (team_sharp_ctx->sharp_context) {
        sharp_coll_finalize(team_sharp_ctx->sharp_context);
    }
    free(team_sharp_ctx);
    return TCCL_OK;
}

static tccl_status_t
tccl_team_sharp_create_post(tccl_team_context_h team_context,
                           tccl_team_config_h config,
                           tccl_oob_collectives_t oob,
                           tccl_team_h *team)
{
    tccl_team_sharp_context_t *team_sharp_ctx =
        (tccl_team_sharp_context_t *)team_context;
    tccl_team_sharp_t *team_sharp             = malloc(sizeof(*team_sharp));
    struct sharp_coll_comm_init_spec comm_spec;
    int i;
    
    team_sharp->oob = oob;
    team_sharp->super.ctx = team_context;

    comm_spec.size              = oob.size;
    comm_spec.rank              = oob.rank;
    comm_spec.group_world_ranks = NULL;
    comm_spec.oob_ctx           = &team_sharp->oob;
    
    int ret = sharp_coll_comm_init(team_sharp_ctx->sharp_context, &comm_spec,
                                   &team_sharp->sharp_comm);
    if (ret<0) {
        if (team_sharp->oob.rank == 0) {
            fprintf(stderr, "SHARP group create failed:%s(%d)",
                    sharp_coll_strerror(ret), ret);
        }
        free(team_sharp);
        return TCCL_ERR_NO_MESSAGE;
    }
    for(i = 0; i < TCCL_TEAM_SHARP_REG_BUF_NUM; i++) {
        int rc;
        team_sharp->bufs[i].buf = malloc(2*TCCL_TEAM_SHARP_REG_BUF_SIZE);
        rc = sharp_coll_reg_mr(team_sharp_ctx->sharp_context,
                               team_sharp->bufs[i].buf,
                               2*TCCL_TEAM_SHARP_REG_BUF_SIZE,
                               &team_sharp->bufs[i].mr);
        if (rc != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "SHARP regmr failed\n");
        }
        team_sharp->bufs[i].used = 0;
    }
    *team = (tccl_team_h)team_sharp;
    return TCCL_OK;
}

static tccl_status_t tccl_team_sharp_destroy(tccl_team_h team)
{
    tccl_team_sharp_t *team_sharp = (tccl_team_sharp_t*)team;
    tccl_team_sharp_context_t *team_sharp_ctx =
        (tccl_team_sharp_context_t *)team_sharp->super.ctx;

    sharp_coll_comm_destroy(team_sharp->sharp_comm);
    for(int i = 0; i < TCCL_TEAM_SHARP_REG_BUF_NUM; i++) {
        int rc;
        rc = sharp_coll_dereg_mr(team_sharp_ctx->sharp_context,
                                 team_sharp->bufs[i].mr);
        if (rc != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "SHARP deregmr failed\n");
        }
        free(team_sharp->bufs[i].buf);
    }
    free(team);
    return TCCL_OK;
}

tccl_team_lib_sharp_t tccl_team_lib_sharp = {
    .super.name                 = "sharp",
    .super.priority             = 100,
    .super.config.reproducible  = TCCL_LIB_NON_REPRODUCIBLE,
    .super.config.thread_mode   = TCCL_LIB_THREAD_SINGLE | TCCL_LIB_THREAD_MULTIPLE,
    .super.config.team_usage    = TCCL_USAGE_HW_COLLECTIVES,
    .super.config.coll_types    = TCCL_BARRIER | TCCL_ALLREDUCE,
    .super.ctx_create_mode      = TCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL,
    .super.create_team_context  = tccl_team_sharp_create_context,
    .super.destroy_team_context = tccl_team_sharp_destroy_context,
    .super.team_create_post     = tccl_team_sharp_create_post,
    .super.team_destroy         = tccl_team_sharp_destroy,
    .super.progress             = NULL,
    .super.collective_init      = tccl_team_sharp_collective_init,
    .super.collective_post      = tccl_team_sharp_collective_post,
    .super.collective_wait      = tccl_team_sharp_collective_wait,
    .super.collective_test      = tccl_team_sharp_collective_test,
    .super.collective_finalize  = tccl_team_sharp_collective_finalize
};
