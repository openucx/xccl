#ifndef XCCL_PROGRESS_QUEUE
#define XCCL_PROGRESS_QUEUE
#include <api/xccl.h>
#include <string.h>

typedef struct xccl_tl_context xccl_tl_context_t;
typedef struct ucc_coll_task ucc_coll_task_t;
typedef struct xccl_progress_queue xccl_progress_queue_t;

typedef struct progress_queue_api{
    xccl_status_t (*progress_queue_enqueue)(xccl_progress_queue_t*, ucc_coll_task_t*);
    xccl_status_t (*progress_queue_progress_tasks)(xccl_progress_queue_t*);
    xccl_status_t (*progress_queue_destroy)(xccl_progress_queue_t*);
} progress_queue_api_t;

struct xccl_progress_queue {
    void* ctx;
    progress_queue_api_t api;
};


xccl_status_t xccl_ctx_progress_queue_init(xccl_progress_queue_t **q, unsigned thread_mode);
xccl_status_t xccl_task_enqueue(xccl_progress_queue_t *q, ucc_coll_task_t *task);
xccl_status_t xccl_ctx_progress_queue(xccl_tl_context_t *tl_ctx);
xccl_status_t xccl_ctx_progress_queue_destroy(xccl_progress_queue_t *q);
#endif
