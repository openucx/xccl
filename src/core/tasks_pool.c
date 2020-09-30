#include "tasks_pool.h"

xccl_status_t tasks_pool_init(xccl_progress_queue_t* handle) {
    handle->ctx = (void*) malloc(sizeof(xccl_tasks_pool_t));
    xccl_tasks_pool_t* ctx = (xccl_tasks_pool_t*) handle->ctx;
    ctx->deque_size = (xccl_lib_global_config.tasks_pool_size / LINE_SIZE) + 1;
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
    ucs_spinlock_init(&(ctx->increase_pool_locks[0]), 0);
    ucs_spinlock_init(&(ctx->increase_pool_locks[1]), 0);
    ucs_spinlock_init(&(ctx->linked_list_lock), 0);
    ctx->linked_list_head              = NULL;
    ctx->linked_list_tail              = NULL;
    ctx->which_pool                    = 0;
    handle->api.progress_queue_enqueue = &tasks_pool_insert;
    handle->api.progress_queue_dequeue = &tasks_pool_pop;
    handle->api.progress_queue_cleanup = &tasks_pool_cleanup;
    return XCCL_OK;
}

xccl_status_t increase_tasks(xccl_tasks_pool_t *ctx, int i) {
    ucs_spin_lock(&ctx->increase_pool_locks[i / ctx->deque_size]);
    if (ctx->tasks[i] == 0) {
        ctx->tasks[i] = (ucc_coll_task_t **) calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
        if (ctx->tasks[i] == NULL) {
            return XCCL_ERR_NO_MEMORY;
        }
    }
    ucs_spin_unlock(&ctx->increase_pool_locks[i / ctx->deque_size]);
    return XCCL_OK;
}

xccl_status_t tasks_pool_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task) {
    xccl_tasks_pool_t *ctx = (xccl_tasks_pool_t*) handle->ctx;
    int i, j;
    int which_pool = task->was_progressed ^(ctx->which_pool & 1);
    int iteration_start = which_pool * ctx->deque_size;
    for (i = iteration_start; i < (iteration_start + ctx->deque_size); i++) {
        if (ctx->tasks[i] == 0) {
            xccl_status_t status = increase_tasks(ctx, i);
            if (status != XCCL_OK) {
                return status;
            }
            return tasks_pool_insert(handle, task);
        }
        for (j = 0; j < LINE_SIZE; j++) {
            /* TODO: Change atomics to UCS once release v1.9.1 is out */
            if (__sync_bool_compare_and_swap(&(ctx->tasks[i][j]), 0, task)) {
                return XCCL_OK;
            }
        }
    }
    ucs_spin_lock(&ctx->linked_list_lock);
    if (ctx->linked_list_head == NULL) {
        ctx->linked_list_head = task;
    } else {
        ctx->linked_list_tail->next = task;
    }
    ctx->linked_list_tail = task;
    task->next = NULL;
    ucs_spin_unlock(&ctx->linked_list_lock);
    return XCCL_OK;
}

xccl_status_t tasks_pool_pop(xccl_progress_queue_t *handle, ucc_coll_task_t **popped_task_ptr, int is_first_call) {
    xccl_tasks_pool_t *ctx = (xccl_tasks_pool_t*) handle->ctx;
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
                return tasks_pool_pop(handle, popped_task_ptr, 0);
            }
            *popped_task_ptr = popped_task;
            return XCCL_OK;
        }
    }
    ucs_spin_lock(&ctx->linked_list_lock);
    if (ctx->linked_list_head != NULL) {
        popped_task = ctx->linked_list_head;
        ctx->linked_list_head = ctx->linked_list_head->next;
    }
    ucs_spin_unlock(&ctx->linked_list_lock);
    if ((popped_task == NULL) && is_first_call) {
        /* TODO: Change atomics to UCS once release v1.9.1 is out */
        __sync_bool_compare_and_swap(&(ctx->which_pool), curr_which_pool, curr_which_pool + 1);
        return tasks_pool_pop(handle, popped_task_ptr, 0);
    }
    *popped_task_ptr = popped_task;
    popped_task->was_progressed = 1;
    return XCCL_OK;
}

xccl_status_t tasks_pool_cleanup(xccl_progress_queue_t *handle) {
    xccl_tasks_pool_t *ctx = (xccl_tasks_pool_t*) handle->ctx;
    int i;
    for (i = 0; i < 2 * ctx->deque_size; i++) {
        free(ctx->tasks[i]);
    }
    free(ctx->tasks);
    ucs_spinlock_destroy(&ctx->increase_pool_locks[0]);
    ucs_spinlock_destroy(&ctx->increase_pool_locks[1]);
    ucs_spinlock_destroy(&ctx->linked_list_lock);
    free(ctx);
    return XCCL_OK;
}
