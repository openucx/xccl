#include "xccl_tasks_queue.h"


xccl_status_t tasks_queue_init(xccl_progress_queue_t *handle) {
    handle->ctx = (void *) malloc(sizeof(xccl_tasks_queue_t));
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    ctx->linked_list_head.next = NULL;
    ctx->linked_list_tail.next = &(ctx->linked_list_head);

    handle->api.progress_queue_enqueue        = &tasks_queue_insert;
    handle->api.progress_queue_progress_tasks = &tasks_queue_progress;
    handle->api.progress_queue_cleanup        = &tasks_queue_cleanup;
    return XCCL_OK;
}

xccl_status_t tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    ctx->linked_list_tail.next->next = task;
    ctx->linked_list_tail.next       = task;
    task->next                       = NULL;
    return XCCL_OK;
}

xccl_status_t tasks_queue_delete(ucc_coll_task_t *prev, ucc_coll_task_t *curr) {
    prev->next = curr->next;

}

xccl_status_t tasks_queue_progress(xccl_progress_queue_t *handle) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    ucc_coll_task_t *curr, *prev;
    if (ctx->linked_list_head.next == NULL) {
        return XCCL_OK;
    }
    for (curr = ctx->linked_list_head.next, prev = &(ctx->linked_list_head); curr != NULL; curr = curr->next){
        if (curr->progress(curr) == XCCL_OK) {
            tasks_queue_delete(prev, curr);
        } else {
            prev = curr;
        }
    }
    ctx->linked_list_tail.next = prev;
    return XCCL_OK;
}

xccl_status_t tasks_queue_cleanup(xccl_progress_queue_t *handle) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    free(ctx);
    return XCCL_OK;
}
