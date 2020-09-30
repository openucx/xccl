/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef TASKS_POOL_H_
#define TASKS_POOL_H_

#include <malloc.h>
#include <ucs/type/spinlock.h>
#include "xccl_global_opts.h"
#include "xccl_schedule.h"
#include "xccl_progress_queue.h"


#define LINE_SIZE 256

extern xccl_config_t xccl_lib_global_config;

typedef struct xccl_tasks_pool {
    ucs_spinlock_t     increase_pool_locks[2];
    ucs_spinlock_t     linked_list_lock;
    ucc_coll_task_t*** tasks;
    unsigned int       which_pool;
    ucc_coll_task_t*   linked_list_head;
    ucc_coll_task_t*   linked_list_tail;
    int                deque_size;
} xccl_tasks_pool_t;

xccl_status_t tasks_pool_init(xccl_progress_queue_t *handle);

xccl_status_t tasks_pool_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task);

xccl_status_t tasks_pool_pop(xccl_progress_queue_t *handle, ucc_coll_task_t **popped_task_ptr, int is_first_call);

xccl_status_t tasks_pool_cleanup(xccl_progress_queue_t *handle);

#endif