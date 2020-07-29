#ifndef XCCL_SCHEDULE_H
#define XCCL_SCHEDULE_H

#include <api/xccl.h>
#include <string.h>

#if defined(CENTRAL_PROGRESS) && !defined(PTRARRAY) && !defined(LFDSA)
#error DEFINE EITHER "PTRARRAY" or "LFDSA"
#endif

#define MAX_LISTENERS 16

typedef enum {
    UCC_EVENT_PROGRESS = 0,
    UCC_EVENT_COMPLETED,
    UCC_EVENT_LAST
} ucc_event_t;

typedef enum {
    UCC_TASK_STATE_NOT_READY,
    UCC_TASK_STATE_INPROGRESS,
    UCC_TASK_STATE_COMPLETED
} ucc_task_state_t;

typedef enum {
    /* Operation completed successfully */
    UCC_OK                              =   0,

    /* Operation is posted and is in progress */
    UCC_INPROGRESS                      =   1,

    /* Operation initialized but not posted */
    UCC_OPERATION_INITIALIZED           =   2,
    UCC_ERR_OP_NOT_SUPPORTED            =   3,
    UCC_ERR_NOT_IMPLEMENTED             =   4,
    UCC_ERR_INVALID_PARAM               =   5,
    UCC_ERR_NO_MEMORY                   =   6,
    UCC_ERR_NO_RESOURCE                 =   7,

    UCC_ERR_LAST                        = -100,
} ucc_status_t;

typedef struct ucc_coll_task ucc_coll_task_t;

typedef void (*ucc_task_event_handler_p)(ucc_coll_task_t *task);

typedef struct ucc_event_manager {
    ucc_coll_task_t  *listeners[UCC_EVENT_LAST][MAX_LISTENERS];
    int listeners_size[UCC_EVENT_LAST];
} ucc_event_manager_t;

typedef struct ucc_coll_task {
    ucc_event_manager_t      em;
    ucc_task_state_t         state;
    ucc_task_event_handler_p handlers[UCC_EVENT_LAST];
} ucc_coll_task_t;

typedef struct xccl_tl_context xccl_tl_context_t;
typedef struct ucc_schedule {
    ucc_coll_task_t    super;
    int                n_completed_tasks;
    xccl_tl_context_t  *tl_ctx;
    volatile int busy;
} ucc_schedule_t;

void ucc_event_manager_init(ucc_event_manager_t *em);
void ucc_event_manager_subscribe(ucc_event_manager_t *em,
                                 ucc_event_t event,
                                 ucc_coll_task_t *task);
void ucc_event_manager_notify(ucc_event_manager_t *em, ucc_event_t event);
void ucc_coll_task_init(ucc_coll_task_t *task);
void schedule_completed_handler(ucc_coll_task_t *task);
void ucc_schedule_init(ucc_schedule_t *schedule, xccl_tl_context_t *tl_ctx);
void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task);
void ucc_schedule_start(ucc_schedule_t *schedule);
ucc_status_t ucc_schedule_progress(ucc_schedule_t *schedule);


#endif
