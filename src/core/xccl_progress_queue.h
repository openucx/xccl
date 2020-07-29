#ifndef XCCL_PROGRESS_QUEUE
#define XCCL_PROGRESS_QUEUE
#include <api/xccl.h>
#include <string.h>

typedef struct xccl_tl_context xccl_tl_context_t;
typedef struct xccl_progress_queue xccl_progress_queue_t;
typedef struct ucc_schedule ucc_schedule_t;
xccl_status_t xccl_ctx_progress_queue_init(xccl_progress_queue_t **q);
xccl_status_t xccl_schedule_enqueue(xccl_progress_queue_t *q,
                                    ucc_schedule_t *schedule);
xccl_status_t xccl_ctx_progress_queue(xccl_tl_context_t *tl_ctx);
#endif
