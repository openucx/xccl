#include <xccl_schedule.h>
#include <xccl_team_lib.h>
#include <xccl_progress_queue.h>

void ucc_event_manager_init(ucc_event_manager_t *em)
{
    int i;
    for (i = 0; i < UCC_EVENT_LAST; i++) {
        em->listeners_size[i] = 0;
    }
}

void ucc_event_manager_subscribe(ucc_event_manager_t *em,
                                 ucc_event_t event,
                                 ucc_coll_task_t *task)
{
    em->listeners[event][em->listeners_size[event]] = task;
    em->listeners_size[event]++;
}


void ucc_event_manager_notify(ucc_event_manager_t *em,
                              ucc_event_t event)
{
    ucc_coll_task_t *task;
    int i;

    for (i = 0; i < em->listeners_size[event]; i++) {
        task = em->listeners[event][i];
        task->handlers[event](task);
    }
}

void ucc_coll_task_init(ucc_coll_task_t *task)
{
    task->state = UCC_TASK_STATE_NOT_READY;
    ucc_event_manager_init(&task->em);
    task->busy = 0;
}

void schedule_completed_handler(ucc_coll_task_t *task)
{
    ucc_schedule_t *self = (ucc_schedule_t*)task;
    self->n_completed_tasks += 1;
    if (self->n_completed_tasks == self->n_tasks) {
        self->super.state = UCC_TASK_STATE_COMPLETED;
    }
}

void ucc_schedule_init(ucc_schedule_t *schedule, xccl_tl_context_t *tl_ctx)
{
    ucc_coll_task_init(&schedule->super);
    schedule->super.handlers[UCC_EVENT_COMPLETED] = schedule_completed_handler;
    schedule->n_completed_tasks = 0;
    schedule->tl_ctx = tl_ctx;
    schedule->n_tasks = 0;
}


void ucc_schedule_add_task(ucc_schedule_t *schedule, ucc_coll_task_t *task)
{
    ucc_event_manager_subscribe(&task->em, UCC_EVENT_COMPLETED, &schedule->super);
    task->schedule = schedule;
    schedule->n_tasks++;
}

void ucc_schedule_start(ucc_schedule_t *schedule)
{
    schedule->super.state = UCC_TASK_STATE_INPROGRESS;
    ucc_event_manager_notify(&schedule->super.em, UCC_EVENT_SCHEDULE_STARTED);
}
