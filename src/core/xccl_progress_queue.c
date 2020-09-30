#include <xccl_progress_queue.h>
#include <xccl_team_lib.h>
#include <xccl_schedule.h>
#include <tasks_pool.h>
#include <tasks_queue.h>


xccl_status_t xccl_ctx_progress_queue_init(xccl_progress_queue_t **q, unsigned thread_mode) {
    xccl_progress_queue_t *pq = (xccl_progress_queue_t*) malloc(sizeof(xccl_progress_queue_t));
    xccl_status_t status;
    switch (thread_mode) {
        case XCCL_THREAD_MODE_SINGLE:
            status = tasks_queue_init(pq);
            break;
        case XCCL_THREAD_MODE_MULTIPLE:
            status = tasks_pool_init(pq);
            break;
        default:
            status = tasks_queue_init(pq);
            break;
    }
    if (status != XCCL_OK) {
        return status;
    }
    *q = pq;
    return XCCL_OK;
}

xccl_status_t xccl_task_enqueue(xccl_progress_queue_t *q,
                                ucc_coll_task_t *task) {
    task->was_progressed = 0;
    return (q->api.progress_queue_enqueue)(q,task);
}

xccl_status_t xccl_ctx_progress_queue(xccl_tl_context_t *tl_ctx) {
    ucc_coll_task_t *popped_task;
    xccl_status_t status = (tl_ctx->pq->api.progress_queue_dequeue)(tl_ctx->pq,&popped_task,1);
    if (status != XCCL_OK) {
        return status;
    }
    if (popped_task) {
        if (XCCL_OK != popped_task->progress(popped_task)) {
            return (tl_ctx->pq->api.progress_queue_enqueue)(tl_ctx->pq,popped_task);
        }
    }
    return XCCL_OK;
}

xccl_status_t xccl_ctx_progress_queue_destroy(xccl_progress_queue_t *q) {
    xccl_status_t status = (q->api.progress_queue_cleanup)(q);
    if (status != XCCL_OK) {
        return status;
    }
    free(q);
    return XCCL_OK;
}
