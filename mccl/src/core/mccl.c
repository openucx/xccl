/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#define _GNU_SOURCE
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

#include "mccl.h"
#include "mccl_core.h"
#include "mccl_team.h"

unsigned long hash(const char *str) {
    unsigned long hash = 5381;
    int c;
    while (c = *str++) {
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
    }
    return hash;
}

static int
oob_tccl_allgather(void *sbuf, void *rbuf, size_t len, void *coll_context) {
    sbgp_t *sbgp = (sbgp_t*)coll_context;
    mccl_comm_t *mccl_comm = sbgp->mccl_comm;
    mccl_comm->config.allgather(sbuf, rbuf, len, mccl_comm->config.comm_rank,
                               sbgp->mccl_rank_map, sbgp->group_size,
                               mccl_comm->config.oob_coll_ctx);
    return 0;
}

static inline char* tccl_libtype_to_char(mccl_tccl_team_lib_t libtype) {
    switch (libtype) {
    case TCCL_LIB_UCX:
        return "ucx";
    case TCCL_LIB_SHMSEG:
        return "shmseg";
    case TCCL_LIB_SHARP:
        return "sharp";
    case TCCL_LIB_VMC:
        return "vmc";
    default:
        break;
    }
    return NULL;
}

static mccl_status_t
mccl_tccl_init_lib(mccl_context_t *ctx, mccl_tccl_team_lib_t libtype) {
    if (!ctx->libs[libtype].enabled) {
        return MCCL_SUCCESS;
    }

    tccl_context_config_t team_ctx_config = {
        .field_mask = TCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME |
                      TCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE   |
                      TCCL_CONTEXT_CONFIG_FIELD_OOB           |
                      TCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE,
        .team_lib_name    = tccl_libtype_to_char(libtype),
        .thread_mode      = TCCL_LIB_THREAD_SINGLE,
        .completion_type  = TCCL_TEAM_COMPLETION_BLOCKING,
        .oob.allgather    = ctx->config.allgather,
        .oob.coll_context = ctx->config.oob_coll_ctx,
        .oob.rank         = ctx->config.world_rank,
        .oob.size         = ctx->config.world_size
    };
    if (TCCL_OK != tccl_create_context(ctx->tccl_lib, team_ctx_config,
                                       &ctx->libs[libtype].tccl_ctx)) {
        return MCCL_ERROR;
    }
    return MCCL_SUCCESS;
}

static mccl_status_t init_env_params(mccl_context_t *ctx)
{
    char *var;
    /* Just quick simple getenv to have smth working.
       TODO We need a parameter registration framework. */

    var = getenv("MCCL_ENABLE_SHARP");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->libs[TCCL_LIB_SHARP].enabled = 1;
    } else {
        ctx->libs[TCCL_LIB_SHARP].enabled = 0;
    }

    var = getenv("MCCL_ENABLE_SHMSEG");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->libs[TCCL_LIB_SHMSEG].enabled = 1;
    } else {
        ctx->libs[TCCL_LIB_SHMSEG].enabled = 0;
    }

    var = getenv("MCCL_ENABLE_VMC");
    if (var && (0 == strcmp(var, "y") ||
                0 == strcmp(var, "1"))) {
        ctx->libs[TCCL_LIB_VMC].enabled = 1;
    } else {
        ctx->libs[TCCL_LIB_VMC].enabled = 0;
    }

    var = getenv("MCCL_BCAST_PIPELINE_THRESH");
    if (var) {
        if (0 == strcmp("inf", var) || 0 == strcmp("INF", var)) {
            ctx->bcast_pipeline_thresh = SIZE_MAX;
        } else {
            ctx->bcast_pipeline_thresh = (size_t)atoi(var);
        }
    } else {
        ctx->bcast_pipeline_thresh = SIZE_MAX;
    }
    return MCCL_SUCCESS;
}

mccl_status_t mccl_init_context(mccl_config_t *conf, mccl_context_h *context) {
    mccl_context_t *ctx;
    char hostname[256];
    int i;
    ctx = (mccl_context_t*)malloc(sizeof(*ctx));
    memcpy(&ctx->config, conf, sizeof(*conf));
    gethostname(hostname, sizeof(hostname));
    ctx->local_proc.node_hash = hash(hostname);
    ctx->procs = (proc_data_t*)malloc(conf->world_size*sizeof(proc_data_t));
    mccl_get_bound_socket_id(&ctx->local_proc.socketid);
    memset(ctx->libs, 0, sizeof(ctx->libs));
    *context = NULL;
    for (i=0; i<TCCL_LIB_LAST; i++) {
        ctx->libs[i].enabled = 1;
    }
    init_env_params(ctx);

    tccl_lib_config_t lib_config = {
        .field_mask = TCCL_LIB_CONFIG_FIELD_TEAM_USAGE,
        .team_usage = TCCL_USAGE_SW_COLLECTIVES |
        TCCL_USAGE_HW_COLLECTIVES,
    };

    if (TCCL_OK != tccl_lib_init(lib_config, &ctx->tccl_lib)) {
        return MCCL_ERROR;
    }

    /* printf("node: %s, node_hash %u, scoket id %d\n", hostname, */
       /* ctx->local_proc.node_hash, ctx->local_proc.socketid); */
    for (i=0; i<TCCL_LIB_LAST; i++) {
        if (MCCL_SUCCESS != mccl_tccl_init_lib(ctx, i)) {
            return MCCL_ERROR;
        }
    }

    *((mccl_context_t**)context) = ctx;
    return MCCL_SUCCESS;
}

mccl_status_t mccl_finalize(mccl_context_h context) {
    mccl_context_t *ctx = (mccl_context_t *)context;
    int i;
    for (i=0; i<TCCL_LIB_LAST; i++) {
        if (ctx->libs[i].enabled) {
            assert(ctx->libs[i].tccl_ctx);
            tccl_destroy_context(ctx->libs[i].tccl_ctx);
        }
    }
    tccl_lib_finalize(ctx->tccl_lib);
    free(ctx->procs);
    free(ctx);
    return MCCL_SUCCESS;
}

static int compare_ints(const void* a, const void* b) {
    return *((int*)a) - *((int*)b);
}

static int compare_proc_data(const void* a, const void* b) {
    const proc_data_t *d1 = (const proc_data_t*)a;
    const proc_data_t *d2 = (const proc_data_t*)b;
    if (d1->node_hash != d2->node_hash) {
        return d1->node_hash > d2->node_hash ? 1 : -1;
    } else {
        return d1->socketid - d2->socketid;
    }
}

static void compute_layout(mccl_context_t *ctx) {
    proc_data_t *sorted = (proc_data_t*)malloc(ctx->config.world_size*sizeof(proc_data_t));
    memcpy(sorted, ctx->procs, ctx->config.world_size*sizeof(proc_data_t));
    qsort(sorted, ctx->config.world_size, sizeof(proc_data_t), compare_proc_data);
    unsigned long current_hash = sorted[0].node_hash;
    int current_ppn = 1;
    int min_ppn = INT_MAX;
    int max_ppn = 0;
    int nnodes = 1;
    int i, j;
    for (i=1; i<ctx->config.world_size; i++) {
        unsigned long hash = sorted[i].node_hash;
        if (hash != current_hash) {
            for (j=0; j<ctx->config.world_size; j++) {
                if (ctx->procs[j].node_hash == current_hash) {
                    ctx->procs[j].node_id = nnodes - 1;
                }
            }
            if (current_ppn > max_ppn) max_ppn = current_ppn;
            if (current_ppn < min_ppn) min_ppn = current_ppn;
            nnodes++;
            current_hash = hash;
            current_ppn = 1;
        } else {
            current_ppn++;
        }
    }
    for (j=0; j<ctx->config.world_size; j++) {
        if (ctx->procs[j].node_hash == current_hash) {
            ctx->procs[j].node_id = nnodes - 1;
        }
    }

    if (current_ppn > max_ppn) max_ppn = current_ppn;
    if (current_ppn < min_ppn) min_ppn = current_ppn;
    free(sorted);
    ctx->nnodes = nnodes;
    ctx->min_ppn = min_ppn;
    ctx->max_ppn = max_ppn;
}

static int mccl_team_rank_to_world(int team_rank, void *rank_mapper_ctx) {
    sbgp_t *sbgp = (sbgp_t*)rank_mapper_ctx;
    int mccl_ctx_rank = sbgp->mccl_rank_map[team_rank];
    assert(sbgp->mccl_comm->world_ranks || sbgp->mccl_comm->config.is_world == 1);
    return sbgp->mccl_comm->world_ranks ?
        sbgp->mccl_comm->world_ranks[mccl_ctx_rank] : mccl_ctx_rank;;
}

static mccl_status_t mccl_create_team(sbgp_t *sbgp, mccl_comm_t *comm,
                                      mccl_tccl_team_lib_t libtype, mccl_team_type_t teamtype) {
    mccl_context_t *mccl_ctx = comm->config.mccl_ctx;
    if (sbgp->status != SBGP_ENABLED) {
        return 0;
    }

    tccl_team_config_t team_config = {
        .range.type      = TCCL_EP_RANGE_CB,
        .range.cb.cb     = mccl_team_rank_to_world,
        .range.cb.cb_ctx = (void*)sbgp,
    };

    tccl_oob_collectives_t oob = {
        .allgather  = oob_tccl_allgather,
        .coll_context = (void*)sbgp,
        .rank = sbgp->group_rank,
        .size = sbgp->group_size,
    };

    comm->teams[teamtype] = (mccl_team_t*)malloc(sizeof(mccl_team_t));
    tccl_team_create_post(mccl_ctx->libs[libtype].tccl_ctx, &team_config,
                         oob, &comm->teams[teamtype]->tccl_team);
    comm->teams[teamtype]->sbgp = sbgp;
    return MCCL_SUCCESS;
}

mccl_status_t mccl_comm_create(mccl_comm_config_t *conf, mccl_comm_h *mccl_comm) {
    mccl_context_t *ctx = conf->mccl_ctx;
    mccl_comm_t *comm = (mccl_comm_t*)calloc(1, sizeof(*comm));
    int i;

    memcpy(&comm->config, conf, sizeof(*conf));
    comm->seq_num = 1;
    comm->ctx_id = 123; //TODO
    if (conf->is_world) {
        conf->allgather(&ctx->local_proc, ctx->procs,
                        sizeof(proc_data_t), 0, NULL, 0, conf->oob_coll_ctx);
        comm->world_ranks = NULL;
        compute_layout(ctx);
    } else {
        comm->world_ranks = (int*)malloc(conf->comm_size*sizeof(int));
        conf->allgather(&conf->world_rank, comm->world_ranks, sizeof(int),
                        0, NULL, 0, conf->oob_coll_ctx);
        qsort(comm->world_ranks, conf->comm_size, sizeof(int), compare_ints);
        //TODO need to carefully recompute my_rank after re-ordering
    }
    for (i=0; i<SBGP_LAST; i++) {
        comm->sbgps[i].status = SBGP_DISABLED;
    }

    sbgp_create(comm, SBGP_NODE, &comm->sbgps[SBGP_NODE]);
    sbgp_create(comm, SBGP_SOCKET, &comm->sbgps[SBGP_SOCKET]);
    sbgp_create(comm, SBGP_NODE_LEADERS, &comm->sbgps[SBGP_NODE_LEADERS]);
    sbgp_create(comm, SBGP_SOCKET_LEADERS, &comm->sbgps[SBGP_SOCKET_LEADERS]);

    /* mccl_create_team(&comm->sbgps[SBGP_NODE],           comm, TCCL_LIB_UCX,    MCCL_TEAM_NODE_UCX); */

    mccl_create_team(&comm->sbgps[SBGP_SOCKET],         comm, TCCL_LIB_UCX,    MCCL_TEAM_SOCKET_UCX);
    mccl_create_team(&comm->sbgps[SBGP_SOCKET_LEADERS], comm, TCCL_LIB_UCX,    MCCL_TEAM_SOCKET_LEADERS_UCX);
    mccl_create_team(&comm->sbgps[SBGP_NODE_LEADERS],   comm, TCCL_LIB_UCX,    MCCL_TEAM_NODE_LEADERS_UCX);

    if (ctx->libs[TCCL_LIB_SHMSEG].enabled) {
        mccl_create_team(&comm->sbgps[SBGP_SOCKET],         comm, TCCL_LIB_SHMSEG, MCCL_TEAM_SOCKET_SHMSEG);
        mccl_create_team(&comm->sbgps[SBGP_SOCKET_LEADERS], comm, TCCL_LIB_SHMSEG, MCCL_TEAM_SOCKET_LEADERS_SHMSEG);
    }

    if (ctx->libs[TCCL_LIB_SHARP].enabled) {
        mccl_create_team(&comm->sbgps[SBGP_NODE_LEADERS],   comm, TCCL_LIB_SHARP,  MCCL_TEAM_NODE_LEADERS_SHARP);
    }

    if (ctx->libs[TCCL_LIB_VMC].enabled) {
        mccl_create_team(&comm->sbgps[SBGP_NODE_LEADERS],   comm, TCCL_LIB_VMC,  MCCL_TEAM_NODE_LEADERS_VMC);
    }

    *((mccl_comm_t**)mccl_comm) = comm;
    return MCCL_SUCCESS;
}

mccl_status_t mccl_comm_free(mccl_comm_h comm) {
    mccl_comm_t *mccl_comm = (mccl_comm_t *)comm;
    int i;
    for (i=0; i<MCCL_TEAM_LAST; i++) {
        if (mccl_comm->teams[i]) {
            tccl_team_destroy(mccl_comm->teams[i]->tccl_team);
            free(mccl_comm->teams[i]);
        }
    }

    for (i=0; i<SBGP_LAST; i++) {
        if (SBGP_ENABLED == mccl_comm->sbgps[i].status) {
            sbgp_cleanup(&mccl_comm->sbgps[i]);
        }
    }

    if (mccl_comm->world_ranks) {
        free(mccl_comm->world_ranks);
    }

    free(mccl_comm);
    return MCCL_SUCCESS;
}

mccl_status_t mccl_progress(mccl_context_h mccl_ctx) {
    mccl_context_t *ctx = (mccl_context_t*)mccl_ctx;
    int i;
    for (i=0; i<TCCL_LIB_LAST; i++) {
        if (ctx->libs[i].enabled) {
            tccl_context_progress(ctx->libs[i].tccl_ctx);
        }
    }
    return MCCL_SUCCESS;
}
