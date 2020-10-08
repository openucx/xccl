#include "xccl_lock_free_tasks_queue.h"

xccl_status_t lf_tasks_queue_init(xccl_progress_queue_t *handle) {
    handle->ctx = (void *) malloc(sizeof(xccl_lf_tasks_queue_t));
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;

    ctx->tasks = (xccl_coll_task_t ***) calloc(NUM_POOLS, sizeof(xccl_coll_task_t **));
    if (ctx->tasks == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    ctx->tasks[0] = (xccl_coll_task_t **) calloc(LINE_SIZE, sizeof(xccl_coll_task_t *));
    if (ctx->tasks[0] == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    ctx->tasks[1] = (xccl_coll_task_t **) calloc(LINE_SIZE, sizeof(xccl_coll_task_t *));
    if (ctx->tasks[1] == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    ucs_spinlock_init(&(ctx->locked_queue_lock), 0);
    ucs_list_head_init(&ctx->locked_queue);
    ctx->which_pool = 0;
    ctx->tasks_countrs[0] = 0;
    ctx->tasks_countrs[1] = 0;

    handle->api.progress_queue_enqueue        = &lf_tasks_queue_insert;
    handle->api.progress_queue_progress_tasks = &lf_tasks_queue_progress;
    handle->api.progress_queue_destroy        = &lf_tasks_queue_destroy;
    return XCCL_OK;
}


xccl_status_t lf_tasks_queue_insert(xccl_progress_queue_t *handle, xccl_coll_task_t *task) {
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;
    int i, j;
    xccl_status_t status;
    int which_pool = task->was_progressed ^(ctx->which_pool & 1);
    for (i = 0; i < LINE_SIZE; i++) {
        if (__sync_bool_compare_and_swap(&(ctx->tasks[which_pool][i]), 0, task)) {
            ucs_atomic_add32(&ctx->tasks_countrs[which_pool], 1);
            return XCCL_OK;
        }
    }
    ucs_spin_lock(&ctx->locked_queue_lock);
    ucs_list_add_tail(&ctx->locked_queue, &task->list_elem);
    ucs_spin_unlock(&ctx->locked_queue_lock);
    return XCCL_OK;
}

xccl_status_t lf_tasks_queue_progress(xccl_progress_queue_t *handle) {
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;
    xccl_coll_task_t *popped_task;
    xccl_status_t status = lf_tasks_queue_pop(ctx, &popped_task, 1);
    if (status != XCCL_OK) {
        return status;
    }
    if (popped_task) {
        if (XCCL_OK != popped_task->progress(popped_task)) {
            return lf_tasks_queue_insert(handle, popped_task);
        }
    }
    return XCCL_OK;
}

xccl_status_t lf_tasks_queue_pop(xccl_lf_tasks_queue_t *ctx, xccl_coll_task_t **popped_task_ptr, int is_first_call) {
    int i, j;
    int curr_which_pool = ctx->which_pool;
    int which_pool = curr_which_pool & 1;
    xccl_coll_task_t *popped_task = NULL;
    if (ctx->tasks_countrs[which_pool]) {
        for (i = 0; i < LINE_SIZE; i++) {
            popped_task = ctx->tasks[which_pool][i];
            if (popped_task) {
                if (__sync_bool_compare_and_swap(&(ctx->tasks[which_pool][i]), popped_task, 0)) {
                    ucs_atomic_sub32(&ctx->tasks_countrs[which_pool], 1);
                    *popped_task_ptr = popped_task;
                    popped_task->was_progressed = 1;
                    return XCCL_OK;
                } else {
                    i = -1;
                    break;
                }
            }
        }
    }
    if (is_first_call) {
        /* TODO: Change atomics to UCS once release v1.9.1 is out */
        ucs_atomic_cswap32(&ctx->which_pool, curr_which_pool, curr_which_pool + 1);
        return lf_tasks_queue_pop(ctx, popped_task_ptr, 0);
    }
    popped_task = NULL;
    ucs_spin_lock(&ctx->locked_queue_lock);
    if (!ucs_list_is_empty(&ctx->locked_queue)) {
        popped_task = ucs_list_extract_head(&ctx->locked_queue, xccl_coll_task_t, list_elem);
    }
    ucs_spin_unlock(&ctx->locked_queue_lock);
    if (popped_task != NULL) {
        popped_task->was_progressed = 1;
    }
    *popped_task_ptr = popped_task;
    return XCCL_OK;
}

xccl_status_t lf_tasks_queue_destroy(xccl_progress_queue_t *handle) {
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;
    int i;
    for (i = 0; i < NUM_POOLS; i++) {
        free(ctx->tasks[i]);
    }
    free(ctx->tasks);
    ucs_spinlock_destroy(&ctx->locked_queue_lock);
    free(ctx);
    return XCCL_OK;
}
