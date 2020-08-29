#include <malloc.h>
#include "xccl_global_opts.h"
#include "xccl_schedule.h"


#define LINE_SIZE 256

extern xccl_config_t xccl_lib_global_config;

typedef struct context {
    int increase_pool_locks[2];
    int linked_list_locks[2];
    ucc_coll_task_t ***tasks;
    unsigned int which_pool;
    ucc_coll_task_t *linked_lists[2];
    int deque_size;
} context;

xccl_status_t tasks_pool_init(context *ctx);

xccl_status_t tasks_pool_insert(context *ctx, ucc_coll_task_t *task);

xccl_status_t tasks_pool_pop(context *ctx, ucc_coll_task_t **popped_task_ptr, int is_first_call);

xccl_status_t tasks_pool_cleanup(context *ctx);

