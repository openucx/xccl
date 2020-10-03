/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef LF_TASKS_QUEUE_H_
#define LF_TASKS_QUEUE_H_

#include <malloc.h>
#include <ucs/type/spinlock.h>
#include <ucs/datastruct/list.h>
#include "xccl_global_opts.h"
#include "xccl_schedule.h"
#include "xccl_progress_queue.h"
#include "xccl_tasks_queue.h"


#define LINE_SIZE 8

extern xccl_config_t xccl_lib_global_config;

typedef struct xccl_lf_tasks_queue {
    ucs_spinlock_t     increase_queue_locks[2];
    ucs_spinlock_t     locked_queue_lock;
    ucc_coll_task_t*** tasks;
    unsigned int       which_pool;
    ucs_list_link_t    locked_queue;
    int                deque_size;
} xccl_lf_tasks_queue_t;

xccl_status_t lf_tasks_queue_init(xccl_progress_queue_t *handle);

xccl_status_t lf_tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task);

xccl_status_t lf_tasks_queue_progress(xccl_progress_queue_t *handle);

xccl_status_t lf_tasks_queue_cleanup(xccl_progress_queue_t *handle);

#endif