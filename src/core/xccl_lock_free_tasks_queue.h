/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef LF_TASKS_QUEUE_H_
#define LF_TASKS_QUEUE_H_

#include <malloc.h>
#include <ucs/type/spinlock.h>
#include <ucs/arch/atomic.h>
#include <ucs/datastruct/list.h>
#include "xccl_global_opts.h"
#include "xccl_schedule.h"
#include "xccl_progress_queue.h"


#define LINE_SIZE 8
#define NUM_POOLS 2

extern xccl_config_t xccl_lib_global_config;

typedef struct xccl_lf_tasks_queue {
    ucs_spinlock_t     locked_queue_lock;
    ucc_coll_task_t*** tasks;
    uint32_t           which_pool;
    ucs_list_link_t    locked_queue;
    uint32_t           tasks_countrs[2];
} xccl_lf_tasks_queue_t;

xccl_status_t lf_tasks_queue_init(xccl_progress_queue_t *handle);

xccl_status_t lf_tasks_queue_insert(xccl_progress_queue_t *handle, ucc_coll_task_t *task);

xccl_status_t lf_tasks_queue_progress(xccl_progress_queue_t *handle);

xccl_status_t lf_tasks_queue_destroy(xccl_progress_queue_t *handle);

#endif