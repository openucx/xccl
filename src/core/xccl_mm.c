#include <xccl_mm.h>

xccl_status_t xccl_global_mem_map_start(xccl_team_h team, xccl_mem_map_params_t params,
                                        xccl_mem_h *memh_p)
{
    XCCL_CHECK_TEAM(team);
    xccl_status_t status;
    int i;
    xccl_team_lib_t *tl;
    xccl_mem_handle_t *memh = calloc(1, sizeof(*memh) +
                                     sizeof(xccl_tl_mem_h)*(team->n_teams-1));
    for (i=0; i<team->n_teams; i++) {
        tl = team->tl_teams[i]->ctx->lib;
        if (tl->global_mem_map_start) {
            if (XCCL_OK != (status = tl->global_mem_map_start(
                                team->tl_teams[i], params, &memh->handles[i]))) {
                goto error;
            }
            memh->handles[i]->id = tl->id;
        }
    }
    memh->team = team;
    *memh_p = memh;
    return XCCL_OK;
error:
    *memh_p = NULL;
    free(memh);
    return status;
}

xccl_status_t xccl_global_mem_map_test(xccl_mem_h memh_p)
{
    int               all_done = 1;
    xccl_mem_handle_t *memh = memh_p;
    xccl_status_t     status;
    xccl_team_lib_t   *tl;
    int               i;

    for (i=0; i<memh->team->n_teams; i++) {
        tl = memh->team->tl_teams[i]->ctx->lib;
        if (memh->handles[i]) {
            assert(tl->global_mem_map_test);
            status = tl->global_mem_map_test(memh->handles[i]);
            if (XCCL_INPROGRESS == status) {
                all_done = 0;
            } else if (XCCL_OK != status) {
                return status;
            }
        }
    }
    return all_done == 1 ? XCCL_OK : XCCL_INPROGRESS;
}

xccl_status_t xccl_global_mem_unmap(xccl_mem_h memh_p)
{
    xccl_mem_handle_t *memh = memh_p;
    xccl_status_t     status;
    int               i;
    xccl_team_lib_t   *tl;

    for (i=0; i<memh->team->n_teams; i++) {
        tl = memh->team->tl_teams[i]->ctx->lib;
        if (memh->handles[i]) {
            assert(tl->global_mem_unmap);
            if (XCCL_OK != (status = tl->global_mem_unmap(memh->handles[i]))) {
                return status;
            }
        }
    }
    free(memh);
    return XCCL_OK;
}
