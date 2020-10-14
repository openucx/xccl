#include "xccl_tasks_queue.h"

xccl_status_t tasks_queue_init(xccl_progress_queue_t *handle) {
    handle->ctx = (void *) malloc(sizeof(xccl_tasks_queue_t));
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    ucs_list_head_init(&ctx->list);

    handle->api.progress_queue_enqueue        = &tasks_queue_insert;
    handle->api.progress_queue_progress_tasks = &tasks_queue_progress;
    handle->api.progress_queue_destroy        = &tasks_queue_destroy;
    return XCCL_OK;
}

xccl_status_t tasks_queue_insert(xccl_progress_queue_t *handle, xccl_coll_task_t *task) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    ucs_list_add_tail(&ctx->list, &task->list_elem);
    return XCCL_OK;
}

xccl_status_t tasks_queue_progress(xccl_progress_queue_t *handle) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    xccl_coll_task_t *task, *tmp;
    xccl_status_t status;
    ucs_list_for_each_safe(task, tmp, &ctx->list, list_elem)
    {
        if (task->progress) {
            if (0 < task->progress(task)) {
                return status;
            }
        }
        if (XCCL_TASK_STATE_COMPLETED == task->state) {
            xccl_event_manager_notify(&task->em, XCCL_EVENT_COMPLETED);
            ucs_list_del(&task->list_elem);
        }
    }
    return XCCL_OK;
}

xccl_status_t tasks_queue_destroy(xccl_progress_queue_t *handle) {
    xccl_tasks_queue_t *ctx = (xccl_tasks_queue_t *) handle->ctx;
    free(ctx);
    return XCCL_OK;
}
