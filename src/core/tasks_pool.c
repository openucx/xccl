#include "tasks_pool.h"

xccl_status_t tasks_pool_init(context *ctx) {
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
    ctx->increase_pool_locks[0] = 0;
    ctx->increase_pool_locks[1] = 0;
    ctx->linked_list_locks[0] = 0;
    ctx->linked_list_locks[1] = 0;
    ctx->linked_lists[0] = NULL;
    ctx->linked_lists[1] = NULL;
    ctx->which_pool = 0;
    return XCCL_OK;
}

xccl_status_t increase_tasks(context *ctx, int i) {
    while (!__sync_bool_compare_and_swap(&(ctx->increase_pool_locks[i / ctx->deque_size]), 0, 1)) {};
    if (ctx->tasks[i] == 0) {
        ctx->tasks[i] = (ucc_coll_task_t **) calloc(LINE_SIZE, sizeof(ucc_coll_task_t *));
        if (ctx->tasks[i] == NULL) {
            return XCCL_ERR_NO_MEMORY;
        }
    }
    while (!__sync_bool_compare_and_swap(&(ctx->increase_pool_locks[i / ctx->deque_size]), 1, 0)) {};
    return XCCL_OK;
}

xccl_status_t tasks_pool_insert(context *ctx, ucc_coll_task_t *task) {
    int i, j;
    int which_pool = task->was_progressed ^(ctx->which_pool & 1);
    int iteration_start = which_pool * ctx->deque_size;
    for (i = iteration_start; i < (iteration_start + ctx->deque_size); i++) {
        if (ctx->tasks[i] == 0) {
            xccl_status_t status = increase_tasks(ctx, i);
            if (status != XCCL_OK) {
                return status;
            }
            return tasks_pool_insert(ctx, task);
        }
        for (j = 0; j < LINE_SIZE; j++) {
            if (__sync_bool_compare_and_swap(&(ctx->tasks[i][j]), 0, task)) {
                return XCCL_OK;
            }
        }
    }
    while (!__sync_bool_compare_and_swap(&(ctx->linked_list_locks[which_pool]), 0, 1)) {};
    if (ctx->linked_lists[which_pool] == NULL) {
        ctx->linked_lists[which_pool] = task;
        task->next = NULL;
    } else {
        task->next = ctx->linked_lists[which_pool];
        ctx->linked_lists[which_pool] = task;
    }
    while (!__sync_bool_compare_and_swap(&(ctx->linked_list_locks[which_pool]), 1, 0)) {};
    return XCCL_OK;
}

xccl_status_t tasks_pool_pop(context *ctx, ucc_coll_task_t **popped_task_ptr, int is_first_call) {
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
                __sync_bool_compare_and_swap(&(ctx->which_pool), curr_which_pool, curr_which_pool + 1);
                return tasks_pool_pop(ctx, popped_task_ptr, 0);
            }
            *popped_task_ptr = popped_task;
            return XCCL_OK;
        }
    }
    while (!__sync_bool_compare_and_swap(&(ctx->linked_list_locks[which_pool]), 0, 1)) {};
    if (ctx->linked_lists[which_pool] != NULL) {
        popped_task = ctx->linked_lists[which_pool];
        ctx->linked_lists[which_pool] = ctx->linked_lists[which_pool]->next;
    }
    while (!__sync_bool_compare_and_swap(&(ctx->linked_list_locks[which_pool]), 1, 0)) {};
    if ((popped_task == NULL) && is_first_call) {
        __sync_bool_compare_and_swap(&(ctx->which_pool), curr_which_pool, curr_which_pool + 1);
        return tasks_pool_pop(ctx, popped_task_ptr, 0);
    }
    *popped_task_ptr = popped_task;
    popped_task->was_progressed = 1;
    return XCCL_OK;
}

xccl_status_t tasks_pool_cleanup(context *ctx) {
    int i;
    for (i = 0; i < 2 * ctx->deque_size; i++) {
        free(ctx->tasks[i]);
    }
    free(ctx->tasks);
    return XCCL_OK;
}
