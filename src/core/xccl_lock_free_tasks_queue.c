#include "xccl_lock_free_tasks_queue.h"

xccl_status_t lf_tasks_queue_init(xccl_progress_queue_t *handle) {
    handle->ctx = (void *) malloc(sizeof(xccl_lf_tasks_queue_t));
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;

    ctx->deque_size = (xccl_lib_global_config.lf_tasks_queue_size / LINE_SIZE) + 1;

    ctx->tasks = (ucc_coll_task_t ***) calloc(2 * ctx->deque_size, sizeof(ucc_coll_task_t **));
    if (ctx->tasks == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    ctx->tasks[0] = (ucc_coll_task_t **) calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
    if (ctx->tasks[0] == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    ctx->tasks[ctx->deque_size] = (ucc_coll_task_t **) calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
    if (ctx->tasks[ctx->deque_size] == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }

    ucs_spinlock_init(&(ctx->increase_queue_locks[0]), 0);
    ucs_spinlock_init(&(ctx->increase_queue_locks[1]), 0);
    ucs_spinlock_init(&(ctx->locked_queue_lock), 0);
    ucs_list_head_init(&ctx->locked_queue);
    ctx->which_pool = 0;

    handle->api.progress_queue_enqueue        = &lf_tasks_queue_insert;
    handle->api.progress_queue_progress_tasks = &lf_tasks_queue_progress;
    handle->api.progress_queue_cleanup        = &lf_tasks_queue_cleanup;
    return XCCL_OK;
}

xccl_status_t increase_tasks(xccl_lf_tasks_queue_t *ctx, int i) {
    ucs_spin_lock(&ctx->increase_queue_locks[i / ctx->deque_size]);
    if (ctx->tasks[i] == 0) {
        ctx->tasks[i] = (ucc_coll_task_t **) calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
        if (ctx->tasks[i] == NULL) {
            return XCCL_ERR_NO_MEMORY;
        }
    }
    ucs_spin_unlock(&ctx->increase_queue_locks[i / ctx->deque_size]);
    return XCCL_OK;
}

xccl_status_t lf_tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task) {
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;
    int i, j;
    xccl_status_t status;
    int which_pool = task->was_progressed ^(ctx->which_pool & 1);
    int iteration_start = which_pool * ctx->deque_size;
    for (i = iteration_start; i < (iteration_start + ctx->deque_size); i++) {
        if (ctx->tasks[i] == 0) {
            status = increase_tasks(ctx, i);
            if (status != XCCL_OK) {
                return status;
            }
            return lf_tasks_queue_insert(handle, task);
        }
        for (j = 0; j < LINE_SIZE; j++) {
            /* TODO: Change atomics to UCS once release v1.9.1 is out */
            if (__sync_bool_compare_and_swap(&(ctx->tasks[i][j]), 0, task)) {
                return XCCL_OK;
            }
        }
    }
    ucs_spin_lock(&ctx->locked_queue_lock);
    ucs_list_add_tail(&ctx->locked_queue, &task->list_elem);
    ucs_spin_unlock(&ctx->locked_queue_lock);
    return XCCL_OK;
}

xccl_status_t lf_tasks_queue_progress(xccl_progress_queue_t *handle) {
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;
    ucc_coll_task_t *popped_task;
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

xccl_status_t lf_tasks_queue_pop(xccl_lf_tasks_queue_t *ctx, ucc_coll_task_t **popped_task_ptr, int is_first_call) {
    int i, j;
    int curr_which_pool = ctx->which_pool;
    int which_pool = curr_which_pool & 1;
    ucc_coll_task_t *popped_task = NULL;
    int iteration_start = which_pool * ctx->deque_size;
    for (i = iteration_start; i < (iteration_start + ctx->deque_size); i++) {
        if (ctx->tasks[i] != 0) {
            for (j = 0; j < LINE_SIZE; j++) {
                popped_task = ctx->tasks[i][j];
                if (popped_task != 0) {
                    /* TODO: Change atomics to UCS once release v1.9.1 is out */
                    if (__sync_bool_compare_and_swap(&(ctx->tasks[i][j]), popped_task, 0)) {
                        *popped_task_ptr = popped_task;
                        popped_task->was_progressed = 1;
                        return XCCL_OK;
                    } else {
                        i = iteration_start - 1;
                        break;
                    }
                }
            }
        } else {
            /* switch between main pool to secondary pool*/
            if (is_first_call) {
                /* TODO: Change atomics to UCS once release v1.9.1 is out */
                __sync_bool_compare_and_swap(&(ctx->which_pool), curr_which_pool, curr_which_pool + 1);
                return lf_tasks_queue_pop(ctx, popped_task_ptr, 0);
            }
            *popped_task_ptr = popped_task;
            return XCCL_OK;
        }
    }
    popped_task = NULL;
    ucs_spin_lock(&ctx->locked_queue_lock);
    if (!ucs_list_is_empty(&ctx->locked_queue)) {
        popped_task = ucs_list_extract_head(&ctx->locked_queue, ucc_coll_task_t, list_elem);
    }
    ucs_spin_unlock(&ctx->locked_queue_lock);
    if (popped_task == NULL) {
        if (is_first_call) {
            /* TODO: Change atomics to UCS once release v1.9.1 is out */
            __sync_bool_compare_and_swap(&(ctx->which_pool), curr_which_pool, curr_which_pool + 1);
            return lf_tasks_queue_pop(ctx, popped_task_ptr, 0);
        }
    }
    else{
        popped_task->was_progressed = 1;
    }
    *popped_task_ptr = popped_task;
    return XCCL_OK;
}

xccl_status_t lf_tasks_queue_cleanup(xccl_progress_queue_t *handle) {
    xccl_lf_tasks_queue_t *ctx = (xccl_lf_tasks_queue_t *) handle->ctx;
    int i;
    for (i = 0; i < 2 * ctx->deque_size; i++) {
        free(ctx->tasks[i]);
    }
    free(ctx->tasks);
    ucs_spinlock_destroy(&ctx->increase_queue_locks[0]);
    ucs_spinlock_destroy(&ctx->increase_queue_locks[1]);
    ucs_spinlock_destroy(&ctx->locked_queue_lock);
    free(ctx);
    return XCCL_OK;
}
