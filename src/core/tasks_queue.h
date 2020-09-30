/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef TASKS_QUEUE_H_
#define TASKS_QUEUE_H_

#include <malloc.h>
#include "xccl_schedule.h"
#include "xccl_progress_queue.h"

typedef struct xccl_tasks_queue {
    ucc_coll_task_t*         linked_list_head;
    ucc_coll_task_t*         linked_list_tail;
} xccl_tasks_queue_t;

xccl_status_t tasks_queue_init(xccl_progress_queue_t *handle);

xccl_status_t tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task);

xccl_status_t tasks_queue_pop(xccl_progress_queue_t *handle, ucc_coll_task_t **popped_task_ptr, int is_first_call);

xccl_status_t tasks_queue_cleanup(xccl_progress_queue_t *handle);

#endif
