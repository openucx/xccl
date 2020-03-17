/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "xccl_sharp_lib.h"
#include "xccl_sharp_collective.h"
#include "xccl_sharp_map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#include <inttypes.h>

unsigned int xccl_sharp_global_rand_state;

__attribute__((constructor))
static void xccl_sharp_constructor(void) {
    setenv("SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD", "0", 0);
}

static inline int xccl_sharp_rand()
{
    return rand_r(&xccl_sharp_global_rand_state);
}

static inline void xccl_sharp_rand_state_init(unsigned int *state)
{
    struct timeval tval;
    gettimeofday(&tval, NULL);
    *state = (unsigned int)(tval.tv_usec ^ getpid());
}

static inline void xccl_sharp_global_rand_state_init()
{
    xccl_sharp_rand_state_init(&xccl_sharp_global_rand_state);
}

static int xccl_sharp_oob_barrier(void *context)
{
    xccl_oob_collectives_t *oob = (xccl_oob_collectives_t*)context;
    int comm_size = oob->size;
    char *tmp = NULL, c = 'c';
    tmp = (char*)malloc(comm_size * sizeof(char));
    xccl_oob_allgather(&c, tmp, sizeof(char), oob);
    free(tmp);
    return 0;
}

static int xccl_sharp_oob_gather(void *context, int root, void *sbuf,
                                 void *rbuf, int size)
{
    xccl_oob_collectives_t *oob = (xccl_oob_collectives_t*)context;
    int comm_size = oob->size;
    int comm_rank = oob->rank;
    void *tmp = NULL;

    if (comm_rank != root) {
        tmp = malloc(comm_size*size);
        rbuf = tmp;
    }
    xccl_oob_allgather(sbuf, rbuf, size, oob);
    if (tmp) {
        free(tmp);
    }
    return 0;
}

static int xccl_sharp_oob_bcast(void *context, void *buf, int size, int root)
{
    xccl_oob_collectives_t *oob = (xccl_oob_collectives_t*)context;
    int comm_size = oob->size;
    int comm_rank = oob->rank;
    void *tmp;
    tmp = malloc(comm_size*size);
    xccl_oob_allgather(buf, tmp, size, oob);
    if (comm_rank != root) {
        memcpy(buf, (void*)((ptrdiff_t)tmp + root*size), size);
    }
    free(tmp);
    return 0;
}

static xccl_status_t
xccl_sharp_create_context(xccl_team_lib_h lib, xccl_context_config_h config,
                          xccl_tl_context_t **context)
{
    xccl_sharp_context_t *ctx = malloc(sizeof(*ctx));
    struct sharp_coll_init_spec init_spec = {0};
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, config);

    xccl_sharp_global_rand_state_init();
    map_xccl_to_sharp_dtype();
    map_xccl_to_sharp_reduce_op_type();

    init_spec.progress_func                  = NULL;
    init_spec.world_rank                     = config->oob.rank;
    init_spec.world_local_rank               = 0;
    init_spec.world_size                     = config->oob.size;
    init_spec.enable_thread_support          = 1;
    init_spec.group_channel_idx              = 0;
    init_spec.oob_colls.barrier              = xccl_sharp_oob_barrier;
    init_spec.oob_colls.bcast                = xccl_sharp_oob_bcast;
    init_spec.oob_colls.gather               = xccl_sharp_oob_gather;
    init_spec.oob_ctx                        = &ctx->super.cfg->oob;
    init_spec.config                         = sharp_coll_default_config;
    init_spec.config.user_progress_num_polls = 1000000;
    init_spec.config.ib_dev_list             = "mlx5_0:1";
    init_spec.job_id                         = xccl_sharp_rand();
    xccl_sharp_oob_bcast((void*)&ctx->super.cfg->oob, &init_spec.job_id,
                         sizeof(uint64_t), 0);
    int ret = sharp_coll_init(&init_spec, &ctx->sharp_context);
    if (ret < 0 ) {
        if (config->oob.rank == 0) {
            fprintf(stderr, "Failed to initialize SHARP collectives:%s(%d)"
                            "job ID:%" PRIu64"\n",
                            sharp_coll_strerror(ret), ret, init_spec.job_id);
        }
        free(ctx);
        return XCCL_ERR_NO_MESSAGE;
    }
    *context = &ctx->super;
    return XCCL_OK;
}

static xccl_status_t
xccl_sharp_destroy_context(xccl_tl_context_t *context)
{
    xccl_sharp_context_t *team_sharp_ctx =
        xccl_derived_of(context, xccl_sharp_context_t);
    if (team_sharp_ctx->sharp_context) {
        sharp_coll_finalize(team_sharp_ctx->sharp_context);
    }
    free(team_sharp_ctx);
    return XCCL_OK;
}

static xccl_status_t
xccl_sharp_team_create_post(xccl_tl_context_t *context,
                            xccl_team_config_h config,
                            xccl_oob_collectives_t oob,
                            xccl_tl_team_t **team)
{
    xccl_sharp_context_t *team_sharp_ctx =
        xccl_derived_of(context, xccl_sharp_context_t);
    xccl_sharp_team_t *team_sharp = malloc(sizeof(*team_sharp));
    struct sharp_coll_comm_init_spec comm_spec;
    int i, ret;
    XCCL_TEAM_SUPER_INIT(team_sharp->super, context, config, oob);

    comm_spec.size              = oob.size;
    comm_spec.rank              = oob.rank;
    comm_spec.group_world_ranks = NULL;
    comm_spec.oob_ctx           = &team_sharp->super.oob;
    ret = sharp_coll_comm_init(team_sharp_ctx->sharp_context, &comm_spec,
                               &team_sharp->sharp_comm);
    if (ret<0) {
        if (oob.rank == 0) {
            fprintf(stderr, "SHARP group create failed:%s(%d)",
                    sharp_coll_strerror(ret), ret);
        }
        free(team_sharp);
        return XCCL_ERR_NO_MESSAGE;
    }
    for(i = 0; i < XCCL_SHARP_REG_BUF_NUM; i++) {
        team_sharp->bufs[i].buf = malloc(2*XCCL_SHARP_REG_BUF_SIZE);
        ret = sharp_coll_reg_mr(team_sharp_ctx->sharp_context,
                               team_sharp->bufs[i].buf,
                               2*XCCL_SHARP_REG_BUF_SIZE,
                               &team_sharp->bufs[i].mr);
        if (ret != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "SHARP regmr failed\n");
        }
        team_sharp->bufs[i].used = 0;
    }
    *team = &team_sharp->super;
    return XCCL_OK;
}

static xccl_status_t xccl_sharp_team_create_test(xccl_tl_team_t *team)
{
    /*TODO implement true non-blocking */
    return XCCL_OK;
}

static xccl_status_t xccl_sharp_team_destroy(xccl_tl_team_t *team)
{
    xccl_sharp_team_t *team_sharp = xccl_derived_of(team, xccl_sharp_team_t);
    xccl_sharp_context_t *team_sharp_ctx =
        xccl_derived_of(team->ctx, xccl_sharp_context_t);

    sharp_coll_comm_destroy(team_sharp->sharp_comm);
    for(int i = 0; i < XCCL_SHARP_REG_BUF_NUM; i++) {
        int rc;
        rc = sharp_coll_dereg_mr(team_sharp_ctx->sharp_context,
                                 team_sharp->bufs[i].mr);
        if (rc != SHARP_COLL_SUCCESS) {
            fprintf(stderr, "SHARP deregmr failed\n");
        }
        free(team_sharp->bufs[i].buf);
    }
    free(team);
    return XCCL_OK;
}

xccl_team_lib_sharp_t xccl_team_lib_sharp = {
    .super.name                 = "sharp",
    .super.id                   = XCCL_TL_SHARP,
    .super.priority             = 90,
    .super.params.reproducible  = XCCL_LIB_NON_REPRODUCIBLE,
    .super.params.thread_mode   = XCCL_LIB_THREAD_SINGLE | XCCL_LIB_THREAD_MULTIPLE,
    .super.params.team_usage    = XCCL_USAGE_HW_COLLECTIVES,
    .super.params.coll_types    = XCCL_COLL_CAP_BARRIER | XCCL_COLL_CAP_ALLREDUCE,
    .super.ctx_create_mode      = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL,
    .super.create_team_context  = xccl_sharp_create_context,
    .super.destroy_team_context = xccl_sharp_destroy_context,
    .super.team_create_post     = xccl_sharp_team_create_post,
    .super.team_create_test     = xccl_sharp_team_create_test,
    .super.team_destroy         = xccl_sharp_team_destroy,
    .super.progress             = NULL,
    .super.team_lib_open        = NULL,
    .super.collective_init      = xccl_sharp_collective_init,
    .super.collective_post      = xccl_sharp_collective_post,
    .super.collective_wait      = xccl_sharp_collective_wait,
    .super.collective_test      = xccl_sharp_collective_test,
    .super.collective_finalize  = xccl_sharp_collective_finalize,
    .super.global_mem_map_start = NULL,
    .super.global_mem_map_test  = NULL,
    .super.global_mem_unmap     = NULL,
};
