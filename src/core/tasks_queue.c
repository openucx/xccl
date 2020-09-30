#include "tasks_queue.h"

xccl_status_t tasks_queue_init(xccl_progress_queue_t* handle) {
    handle->ctx = (void*) malloc(sizeof(xccl_tasks_queue_t));
    xccl_tasks_queue_t* ctx =(xccl_tasks_queue_t*) handle->ctx;
    ctx->linked_list_head       = NULL;
    ctx->linked_list_tail       = NULL;

    handle->api.progress_queue_enqueue = &tasks_queue_insert;
    handle->api.progress_queue_dequeue = &tasks_queue_pop;
    handle->api.progress_queue_cleanup = &tasks_queue_cleanup;
    return XCCL_OK;
}

xccl_status_t tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t*) handle->ctx;
    if (ctx->linked_list_head == NULL) {
        ctx->linked_list_head = task;
    } else {
        ctx->linked_list_tail->next = task;
    } 
    task->next = NULL;
    ctx->linked_list_tail = task;
    return XCCL_OK;
}

xccl_status_t tasks_queue_pop(xccl_progress_queue_t *handle, ucc_coll_task_t **popped_task_ptr, int is_first_call) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t*) handle->ctx;
    ucc_coll_task_t *popped_task = NULL;
    if (ctx->linked_list_head != NULL) {
        popped_task = ctx->linked_list_head;
        ctx->linked_list_head = ctx->linked_list_head->next;
    }
    *popped_task_ptr = popped_task;
    return XCCL_OK;
}

xccl_status_t tasks_queue_cleanup(xccl_progress_queue_t *handle) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t*) handle->ctx;
    free(ctx);
    return XCCL_OK;
}
