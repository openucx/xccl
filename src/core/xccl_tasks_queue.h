/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef TASKS_QUEUE_H_
#define TASKS_QUEUE_H_

#include <malloc.h>
#include <ucs/datastruct/list.h>
#include "xccl_schedule.h"
#include "xccl_progress_queue.h"

typedef struct xccl_tasks_queue {
    ucs_list_link_t list;
} xccl_tasks_queue_t;

xccl_status_t tasks_queue_init(xccl_progress_queue_t *handle);

xccl_status_t tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task);

xccl_status_t tasks_queue_progress(xccl_progress_queue_t *handle);

xccl_status_t tasks_queue_destroy(xccl_progress_queue_t *handle);

#endif
