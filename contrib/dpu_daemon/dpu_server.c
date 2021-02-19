/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

#include "server_xccl.h"
#include "host_channel.h"

#define MAX_THREADS 128
typedef struct {
    pthread_t id;
    int idx, nthreads;
    dpu_xccl_comm_t comm;
    dpu_hc_t *hc;
    unsigned int itt;
} thread_ctx_t;

/* thread accisble data - split reader/writer */
typedef struct {
    volatile unsigned long g_itt;  /* first cache line */
    volatile unsigned long pad[3]; /* pad to 64bytes */
    volatile unsigned long l_itt;  /* second cache line */
    volatile unsigned long pad2[3]; /* pad to 64 bytes */
} thread_sync_t;
    
static thread_sync_t *thread_sync = NULL;

void *dpu_worker(void *arg)
{
    int i = 0;
    thread_ctx_t *ctx = (thread_ctx_t*)arg;
    xccl_coll_req_h request;
   
    while(1) {
        ctx->itt++;
        if (ctx->idx > 0) {
            while (thread_sync[ctx->idx].g_itt < ctx->itt) {
                /* busy wait */
            }
        }
        else {
            dpu_hc_wait(ctx->hc, ctx->itt);
            for (i = 0; i < ctx->nthreads; i++) {
                thread_sync[i].g_itt++;
            }
        }
    
        int offset, block;
        int count = dpu_hc_get_count(ctx->hc);
        int ready = 0;

        block = count / ctx->nthreads;
        offset = block * ctx->idx;
        if(ctx->idx < (count % ctx->nthreads)) {
            offset += ctx->idx;
            block++;
        } else {
            offset += (count % ctx->nthreads);
        }
        
        xccl_coll_op_args_t coll = {
            .field_mask = 0,
            .coll_type = XCCL_ALLREDUCE,
            .buffer_info = {
                .src_buffer = ctx->hc->mem_segs.put.base + offset * sizeof(int),
                .dst_buffer = ctx->hc->mem_segs.get.base + offset * sizeof(int),
                .len        = block * xccl_dt_size(dpu_hc_get_dtype(ctx->hc)),
            },
            .reduce_info = {
                .dt = dpu_hc_get_dtype(ctx->hc),
                .op = dpu_hc_get_op(ctx->hc),
                .count = block,
            },
            .alg.set_by_user = 0,
            .tag  = 123, //todo
        };
        
        if (coll.reduce_info.op == XCCL_OP_UNSUPPORTED) {
            break;
        }

        XCCL_CHECK(xccl_collective_init(&coll, &request, ctx->comm.team));
        XCCL_CHECK(xccl_collective_post(request));
        while (XCCL_OK != xccl_collective_test(request)) {
            xccl_context_progress(ctx->comm.ctx);
        }
        XCCL_CHECK(xccl_collective_finalize(request));

        thread_sync[ctx->idx].l_itt++;

        if (ctx->idx == 0) {
            while (ready != ctx->nthreads) {
                ready = 0;
                for (i = 0; i < ctx->nthreads; i++) {
                    if (thread_sync[i].l_itt == ctx->itt) {
                        ready++;
                    }
                    else {
                        break;
                    }
                }
            }
    
            dpu_hc_reply(ctx->hc, ctx->itt);
        }
    }
        
    return NULL;
}

int main(int argc, char **argv)
{
    int nthreads = 0, i;
    thread_ctx_t *tctx_pool = NULL;
    dpu_xccl_global_t xccl_glob;
    dpu_hc_t hc_b, *hc = &hc_b;

    if (argc < 2 ) {
        printf("Need thread # as an argument\n");
        return 1;
    }
    nthreads = atoi(argv[1]);
    if (MAX_THREADS < nthreads || 0 >= nthreads) {
        printf("ERROR: bad thread #: %d\n", nthreads);
        return 1;
    }
    printf("DPU daemon: Running with %d threads\n", nthreads);
    tctx_pool = calloc(nthreads, sizeof(*tctx_pool));
    XCCL_CHECK(dpu_xccl_init(argc, argv, &xccl_glob));

//     thread_sync = calloc(nthreads, sizeof(*thread_sync));
    thread_sync = aligned_alloc(64, nthreads * sizeof(*thread_sync));
    memset(thread_sync, 0, nthreads * sizeof(*thread_sync));

    dpu_hc_init(hc);
    dpu_hc_accept(hc);

    for(i = 0; i < nthreads; i++) {
//         printf("Thread %d spawned!\n", i);
        XCCL_CHECK(dpu_xccl_alloc_team(&xccl_glob, &tctx_pool[i].comm));
      
        tctx_pool[i].idx = i;
        tctx_pool[i].nthreads = nthreads;
        tctx_pool[i].hc    = hc;
        tctx_pool[i].itt = 0;

        if (i < nthreads - 1) {
            pthread_create(&tctx_pool[i].id, NULL, dpu_worker,
                           (void*)&tctx_pool[i]);
        }
    }

    /* The final DPU worker is executed in this context */
    dpu_worker((void*)&tctx_pool[i-1]);

    for(i = 0; i < nthreads; i++) {
        if (i < nthreads - 1) {
            pthread_join(tctx_pool[i].id, NULL);
        }
        dpu_xccl_free_team(&xccl_glob, &tctx_pool[i].comm);
//         printf("Thread %d joined!\n", i);
    }

    dpu_xccl_finalize(&xccl_glob);
    return 0;
}
