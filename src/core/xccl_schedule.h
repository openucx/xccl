#ifndef XCCL_SCHEDULE_H
#define XCCL_SCHEDULE_H

#include <api/xccl.h>
#include <string.h>
#include <ucs/datastruct/list.h>

#define MAX_LISTENERS 16

typedef enum {
    XCCL_EVENT_COMPLETED = 0,
    XCCL_EVENT_SCHEDULE_STARTED,
    XCCL_EVENT_LAST
} xccl_event_t;

typedef enum {
    XCCL_TASK_STATE_NOT_READY,
    XCCL_TASK_STATE_INPROGRESS,
    XCCL_TASK_STATE_COMPLETED
} xccl_task_state_t;

typedef struct xccl_coll_task xccl_coll_task_t;

typedef void (*xccl_task_event_handler_p)(xccl_coll_task_t *task);

typedef struct xccl_event_manager {
    xccl_coll_task_t  *listeners[XCCL_EVENT_LAST][MAX_LISTENERS];
    int listeners_size[XCCL_EVENT_LAST];
} xccl_event_manager_t;

typedef struct xccl_coll_task {
    xccl_event_manager_t      em;
    xccl_task_state_t         state;
    xccl_task_event_handler_p handlers[XCCL_EVENT_LAST];
    xccl_status_t (*progress)(struct xccl_coll_task *self);
    struct xccl_schedule *schedule;
    volatile int busy;
    /* used for progress queue */
    xccl_coll_task_t* next;
    ucs_list_link_t  list_elem;
    int              was_progressed;
} xccl_coll_task_t;

typedef struct xccl_tl_context xccl_tl_context_t;
typedef struct xccl_schedule {
    xccl_coll_task_t    super;
    int                n_completed_tasks;
    int                n_tasks;
    xccl_tl_context_t  *tl_ctx;
} xccl_schedule_t;

void xccl_event_manager_init(xccl_event_manager_t *em);
void xccl_event_manager_subscribe(xccl_event_manager_t *em,
                                 xccl_event_t event,
                                 xccl_coll_task_t *task);
void xccl_event_manager_notify(xccl_event_manager_t *em, xccl_event_t event);
void xccl_coll_task_init(xccl_coll_task_t *task);
void schedule_completed_handler(xccl_coll_task_t *task);
void xccl_schedule_init(xccl_schedule_t *schedule, xccl_tl_context_t *tl_ctx);
void xccl_schedule_add_task(xccl_schedule_t *schedule, xccl_coll_task_t *task);
void xccl_schedule_start(xccl_schedule_t *schedule);
xccl_status_t xccl_schedule_progress(xccl_schedule_t *schedule);


#endif
