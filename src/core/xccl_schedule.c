#include <xccl_schedule.h>
#include <xccl_team_lib.h>
#include <xccl_progress_queue.h>

void xccl_event_manager_init(xccl_event_manager_t *em)
{
    int i;
    for (i = 0; i < XCCL_EVENT_LAST; i++) {
        em->listeners_size[i] = 0;
    }
}

void xccl_event_manager_subscribe(xccl_event_manager_t *em,
                                 xccl_event_t event,
                                 xccl_coll_task_t *task)
{
    em->listeners[event][em->listeners_size[event]] = task;
    em->listeners_size[event]++;
}

xccl_status_t xccl_event_manager_notify(xccl_event_manager_t *em,
                              xccl_event_t event)
{
    xccl_coll_task_t *task;
    xccl_status_t status;
    int i;

    for (i = 0; i < em->listeners_size[event]; i++) {
        task = em->listeners[event][i];
        status = task->handlers[event](task);
        if (status != XCCL_OK) {
            return status;
        }
    }
    return XCCL_OK;
}

void xccl_coll_task_init(xccl_coll_task_t *task)
{
    task->state = XCCL_TASK_STATE_NOT_READY;
    xccl_event_manager_init(&task->em);
    task->busy = 0;
}

void schedule_completed_handler(xccl_coll_task_t *task)
{
    xccl_schedule_t *self = (xccl_schedule_t*)task;
    self->n_completed_tasks += 1;
    if (self->n_completed_tasks == self->n_tasks) {
        self->super.state = XCCL_TASK_STATE_COMPLETED;
    }
}

void xccl_schedule_init(xccl_schedule_t *schedule, xccl_tl_context_t *tl_ctx)
{
    xccl_coll_task_init(&schedule->super);
    schedule->super.handlers[XCCL_EVENT_COMPLETED] = schedule_completed_handler;
    schedule->n_completed_tasks = 0;
    schedule->tl_ctx = tl_ctx;
    schedule->n_tasks = 0;
}

void xccl_schedule_add_task(xccl_schedule_t *schedule, xccl_coll_task_t *task)
{
    xccl_event_manager_subscribe(&task->em, XCCL_EVENT_COMPLETED, &schedule->super);
    task->schedule = schedule;
    schedule->n_tasks++;
}

void xccl_schedule_start(xccl_schedule_t *schedule)
{
    schedule->super.state = XCCL_TASK_STATE_INPROGRESS;
    xccl_event_manager_notify(&schedule->super.em, XCCL_EVENT_SCHEDULE_STARTED);
}
