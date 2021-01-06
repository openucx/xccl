/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "xccl_dpu_lib.h"
#include <ucs/memory/memory_type.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <poll.h>
#include <errno.h>
#include <unistd.h>

static ucs_config_field_t xccl_team_lib_dpu_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_team_lib_dpu_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_team_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t xccl_tl_dpu_context_config_table[] = {
    {"", "",
     NULL,
     ucs_offsetof(xccl_tl_dpu_context_config_t, super),
     UCS_CONFIG_TYPE_TABLE(xccl_tl_context_config_table)
    },

    {"SERVER_HOSTNAME", "",
     "Bluefield IP address",
     ucs_offsetof(xccl_tl_dpu_context_config_t, server_hname),
     UCS_CONFIG_TYPE_STRING
    },

    {"SERVER_PORT", "13337",
     "Bluefield DPU port",
     ucs_offsetof(xccl_tl_dpu_context_config_t, server_port),
     UCS_CONFIG_TYPE_UINT
    },

    {"ENABLE", "0",
     "Assume server is running on BF",
     ucs_offsetof(xccl_tl_dpu_context_config_t, use_dpu),
     UCS_CONFIG_TYPE_UINT
    },

    {"HOST_DPU_LIST", "",
     "A host-dpu list used to identify the DPU IP",
     ucs_offsetof(xccl_tl_dpu_context_config_t, host_dpu_list),
     UCS_CONFIG_TYPE_STRING
    },

    {NULL}
};

static xccl_status_t xccl_dpu_lib_open(xccl_team_lib_h self,
                                       xccl_team_lib_config_t *config)
{
    xccl_team_lib_dpu_t        *tl  = ucs_derived_of(self, xccl_team_lib_dpu_t);
    xccl_team_lib_dpu_config_t *cfg = ucs_derived_of(config, xccl_team_lib_dpu_config_t);

    tl->config.super.log_component.log_level = cfg->super.log_component.log_level;
    sprintf(tl->config.super.log_component.name, "%s", "TEAM_DPU");
    xccl_dpu_debug("Team DPU opened");
    if (cfg->super.priority != -1) {
        tl->super.priority = cfg->super.priority;
    }

    return XCCL_OK;
}

#define EXCHANGE_LENGTH_TAG 1ull
#define EXCHANGE_RKEY_TAG 2ull
#define EXCHANGE_ADDR_TAG 3ull

static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    xccl_dpu_error("error handling callback was invoked with status %d (%s)\n",
                    status, ucs_status_string(status));
}

typedef enum {
    UCX_REQUEST_ACTIVE,
    UCX_REQUEST_DONE,
} ucx_request_status_t;

typedef struct ucx_request {
  ucx_request_status_t status;
} ucx_request_t;

static void ucx_req_init(void* request)
{
    ucx_request_t *req = (ucx_request_t*)request;
    req->status = UCX_REQUEST_ACTIVE;
}

static void ucx_req_cleanup(void* request){ }

static int _server_connect(char *hname, uint16_t port)
{
    int sock, n;
    struct sockaddr_in addr;
    struct addrinfo *res, *t;
    struct addrinfo hints = { 0 };
    char service[64];


    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    sprintf(service, "%d", port);
    n = getaddrinfo(hname, service, &hints, &res);

    if (n < 0) {
        xccl_dpu_error("%s:%d: getaddrinfo(): %s for %s:%s\n",
                       __FILE__,__LINE__,
                       gai_strerror(n), hname, service);
        return -1;
    }

    for (t = res; t; t = t->ai_next) {
        sock = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
        if (sock >= 0) {
            if (!connect(sock, t->ai_addr, t->ai_addrlen))
                break;
            close(sock);
            sock = -1;
        }
    }
    freeaddrinfo(res);
    return sock;
}


static xccl_status_t
xccl_dpu_context_create(xccl_team_lib_h lib, xccl_context_params_t *params,
                        xccl_tl_context_config_t *config,
                        xccl_tl_context_t **context)
{
    xccl_dpu_context_t *ctx = malloc(sizeof(*ctx));
    ucp_params_t ucp_params;
    ucp_worker_params_t worker_params;
    ucp_ep_params_t ep_params;
    ucs_status_t status;
    xccl_tl_dpu_context_config_t *cfg;
    char hname[256];
    void *rem_worker_addr;
    size_t rem_worker_addr_size;
    int sockfd = 0, found = 0;

    cfg = ucs_derived_of(config, xccl_tl_dpu_context_config_t);
    XCCL_CONTEXT_SUPER_INIT(ctx->super, lib, params);

#if 0
    if (atoi(getenv("OMPI_COMM_WORLD_RANK")) == 0) {
        strcpy(cfg->server_hname, "thor001");
    }
    else if (atoi(getenv("OMPI_COMM_WORLD_RANK")) == 1) {
        strcpy(cfg->server_hname, "thor002");
    }
    else {
        fprintf(stderr, "error %s", __FUNCTION__);
    }
#endif
    gethostname(hname, sizeof(hname) - 1);
    if (cfg->use_dpu) {
        char *h = calloc(1, 256);
        FILE *fp = NULL;

        if (strcmp(cfg->host_dpu_list,"") != 0) {

            fp = fopen(cfg->host_dpu_list, "r");
            if (fp == NULL) {
                fprintf(stderr, "Unable to open \"%s\", disabling dpu team\n", cfg->host_dpu_list);
                cfg->use_dpu = 0;
                found = 0;
            }
            else {
                while (fscanf(fp,"%s", h) != EOF) {
                    if (strcmp(h, hname) == 0) {
                        found = 1;
                        fscanf(fp, "%s", hname);
                        fprintf(stderr, "DPU <%s> found!\n", hname);
                        break;
                    }
                    memset(h, 0, 256);
                }
            }
            if (!found) {
                cfg->use_dpu = 0;
            }
        }
        else {
            fprintf(stderr, "DPU_ENABLE set, but HOST_LIST not specified. Disabling DPU team!\n");
            cfg->use_dpu = 0;
        }
        free(h);
    }
    else {
        goto err;
    }

    if (!found) {
        goto err;
    }

    xccl_dpu_info("Connecting to %s", hname);
    sockfd = _server_connect(hname, cfg->server_port);

    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask      = UCP_PARAM_FIELD_FEATURES |
                                 UCP_PARAM_FIELD_REQUEST_SIZE |
                                 UCP_PARAM_FIELD_REQUEST_INIT |
                                 UCP_PARAM_FIELD_REQUEST_CLEANUP;
    ucp_params.features        = UCP_FEATURE_TAG |
                                 UCP_FEATURE_RMA;
    ucp_params.request_size    = sizeof(ucx_request_t);
    ucp_params.request_init    = ucx_req_init;
    ucp_params.request_cleanup = ucx_req_cleanup;

    status = ucp_init(&ucp_params, NULL, &ctx->ucp_context);
    if (status != UCS_OK) {
        xccl_dpu_error("failed ucp_init(%s)\n", ucs_status_string(status));
        goto err;
    }

    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(ctx->ucp_context, &worker_params, &ctx->ucp_worker);
     if (status != UCS_OK) {
        xccl_dpu_error("failed ucp_worker_create (%s)\n", ucs_status_string(status));
        goto err_cleanup_context;
    }

    ucp_worker_attr_t attr;
    attr.field_mask    = UCP_WORKER_ATTR_FIELD_ADDRESS |
                         UCP_WORKER_ATTR_FIELD_ADDRESS_FLAGS;
    attr.address_flags = UCP_WORKER_ADDRESS_FLAG_NET_ONLY;
    ucp_worker_query(ctx->ucp_worker, &attr);
    int ret;
    ret = send(sockfd, &attr.address_length, sizeof(&attr.address_length), 0);
    if (ret<0) {
        xccl_dpu_error("send length failed");
    }
    ret = send(sockfd, attr.address, attr.address_length, 0);
    if (ret<0) {
        xccl_dpu_error("send address failed");
    }
    ret = recv(sockfd, &rem_worker_addr_size, sizeof(rem_worker_addr_size), MSG_WAITALL);
    if (ret<0) {
        xccl_dpu_error("recv address length failed");
    }
    rem_worker_addr = malloc(rem_worker_addr_size);
    ret = recv(sockfd, rem_worker_addr, rem_worker_addr_size, MSG_WAITALL);
    if (ret<0) {
        xccl_dpu_error("recv address failed");
    }
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = err_cb;
    ep_params.address          = rem_worker_addr;

    status = ucp_ep_create(ctx->ucp_worker, &ep_params, &ctx->ucp_ep);
    free(attr.address);
    free(rem_worker_addr);
    close(sockfd);
    if (status != UCS_OK) {
        xccl_dpu_error("failed to connect to %s (%s)\n",
                       hname, ucs_status_string(status));
        goto err_cleanup_worker;
    }

    *context = &ctx->super;

    xccl_dpu_debug("context created");
    return XCCL_OK;

err_cleanup_worker:
    ucp_worker_destroy(ctx->ucp_worker);
err_cleanup_context:
    ucp_cleanup(ctx->ucp_context);
err:
    return XCCL_ERR_NO_MESSAGE;
}

static xccl_status_t
xccl_dpu_context_destroy(xccl_tl_context_t *context)
{
    xccl_dpu_context_t *dpu_ctx = ucs_derived_of(context, xccl_dpu_context_t);
    ucp_request_param_t param;
    ucs_status_t status;
    void *close_req;

    param.op_attr_mask  = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags         = UCP_EP_CLOSE_FLAG_FORCE;
    close_req           = ucp_ep_close_nbx(dpu_ctx->ucp_ep, &param);
    if (UCS_PTR_IS_PTR(close_req)) {
        do {
            ucp_worker_progress(dpu_ctx->ucp_worker);
            status = ucp_request_check_status(close_req);
        } while (status == UCS_INPROGRESS);
        ucp_request_free (close_req);
    } else if (UCS_PTR_STATUS(close_req) != UCS_OK) {
        xccl_dpu_error("failed to close ep %p\n", (void *)dpu_ctx->ucp_ep);
        return XCCL_ERR_NO_MESSAGE;
    }
    ucp_worker_destroy(dpu_ctx->ucp_worker);
    ucp_cleanup(dpu_ctx->ucp_context);
    free(dpu_ctx);

    return XCCL_OK;
}

static void send_handler_nbx(void *request, ucs_status_t status,
                             void *user_data) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->status = UCX_REQUEST_DONE;
}

void recv_handler_nbx(void *request, ucs_status_t status,
                      const ucp_tag_recv_info_t *tag_info,
                      void *user_data) {
  ucx_request_t *req = (ucx_request_t*)request;
  req->status = UCX_REQUEST_DONE;
}

static xccl_status_t ucx_req_test(ucx_request_t **req, ucp_worker_h worker) {
    if (*req == NULL) {
        return XCCL_OK;
    }

    if ((*req)->status == UCX_REQUEST_DONE) {
        (*req)->status = UCX_REQUEST_ACTIVE;
        ucp_request_free(*req);
        (*req) = NULL;
        return XCCL_OK;
    }
    ucp_worker_progress(worker);
    return XCCL_INPROGRESS;
}

static xccl_status_t ucx_req_check(ucx_request_t *req) {
    if (UCS_PTR_IS_ERR(req)) {
        xccl_dpu_error("failed to send/recv msg");
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;
}

static xccl_status_t
xccl_dpu_team_create_post(xccl_tl_context_t *context,
                          xccl_team_params_t *params,
                          xccl_tl_team_t **team)
{
    xccl_dpu_context_t *ctx = ucs_derived_of(context, xccl_dpu_context_t);
    xccl_dpu_team_t *dpu_team = malloc(sizeof(*dpu_team));
    ucp_mem_map_params_t mmap_params;
    ucp_request_param_t send_req_param, recv_req_param;
    ucx_request_t *send_req[3], *recv_req[2];
    size_t rem_rkeys_lengths[3];
    uint64_t rem_addresses[3];
    void *ctrl_seg_rkey_buf;
    size_t ctrl_seg_rkey_buf_size;
    size_t total_rkey_size;
    void *rem_rkeys;

    XCCL_TEAM_SUPER_INIT(dpu_team->super, context, params);

    dpu_team->coll_id = 1;
    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mmap_params.address    = (void*)&dpu_team->ctrl_seg;
    mmap_params.length     = sizeof(dpu_team->ctrl_seg);

    ucp_mem_map(ctx->ucp_context, &mmap_params, &dpu_team->ctrl_seg_memh);
    ucp_rkey_pack(ctx->ucp_context, dpu_team->ctrl_seg_memh,
                  &ctrl_seg_rkey_buf,
                  &ctrl_seg_rkey_buf_size);

    send_req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_DATATYPE;
    send_req_param.datatype     = ucp_dt_make_contig(1);
    send_req_param.cb.send      = send_handler_nbx;

    recv_req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_DATATYPE;
    recv_req_param.datatype     = ucp_dt_make_contig(1);
    recv_req_param.cb.recv      = recv_handler_nbx;

    send_req[0] = ucp_tag_send_nbx(ctx->ucp_ep, &mmap_params.address,
                                   sizeof(uint64_t), EXCHANGE_ADDR_TAG,
                                   &send_req_param);
    if (ucx_req_check(send_req[0]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }

    send_req[1] = ucp_tag_send_nbx(ctx->ucp_ep, &ctrl_seg_rkey_buf_size,
                                   sizeof(size_t), EXCHANGE_LENGTH_TAG,
                                   &send_req_param);
    if (ucx_req_check(send_req[1]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    send_req[2] = ucp_tag_send_nbx(ctx->ucp_ep, ctrl_seg_rkey_buf,
                                   ctrl_seg_rkey_buf_size, EXCHANGE_RKEY_TAG,
                                   &send_req_param);
    if (ucx_req_check(send_req[2]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    recv_req[0] = ucp_tag_recv_nbx(ctx->ucp_worker, rem_rkeys_lengths,
                                sizeof(rem_rkeys_lengths),
                                EXCHANGE_LENGTH_TAG, (uint64_t)-1,
                                &recv_req_param);
    if (ucx_req_check(recv_req[0]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    do {
        ucp_worker_progress(ctx->ucp_worker);
    } while((ucx_req_test(&(send_req[0]), ctx->ucp_worker) != XCCL_OK) ||
            (ucx_req_test(&(send_req[1]), ctx->ucp_worker) != XCCL_OK) ||
            (ucx_req_test(&(send_req[2]), ctx->ucp_worker) != XCCL_OK) ||
            (ucx_req_test(&(recv_req[0]), ctx->ucp_worker) != XCCL_OK));

//     fprintf (stderr,"%lu ;%lu %lu\n", rem_rkeys_lengths[0], rem_rkeys_lengths[1], rem_rkeys_lengths[2]);

    ucp_rkey_buffer_release(ctrl_seg_rkey_buf);
    total_rkey_size = rem_rkeys_lengths[0] + rem_rkeys_lengths[1] + rem_rkeys_lengths[2];
    rem_rkeys = malloc(total_rkey_size);
    recv_req[0] = ucp_tag_recv_nbx(ctx->ucp_worker, &rem_addresses,
                                   sizeof(rem_addresses),
                                   EXCHANGE_ADDR_TAG, (uint64_t)-1,
                                   &recv_req_param);
    if (ucx_req_check(recv_req[0]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }

    recv_req[1] = ucp_tag_recv_nbx(ctx->ucp_worker, rem_rkeys,
                                total_rkey_size,
                                EXCHANGE_RKEY_TAG, (uint64_t)-1,
                                &recv_req_param);
    if (ucx_req_check(recv_req[1]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    do {
        ucp_worker_progress(ctx->ucp_worker);
    } while((ucx_req_test(&recv_req[0], ctx->ucp_worker) != XCCL_OK) ||
            (ucx_req_test(&recv_req[1], ctx->ucp_worker) != XCCL_OK));

    dpu_team->rem_ctrl_seg = rem_addresses[0];
    ucp_ep_rkey_unpack(ctx->ucp_ep, rem_rkeys, &dpu_team->rem_ctrl_seg_key);
    dpu_team->rem_data_in = rem_addresses[1];
    ucp_ep_rkey_unpack(ctx->ucp_ep, (void*)((ptrdiff_t)rem_rkeys + rem_rkeys_lengths[0]),
                       &dpu_team->rem_data_in_key);
    dpu_team->rem_data_out = rem_addresses[2];
    ucp_ep_rkey_unpack(ctx->ucp_ep, (void*)((ptrdiff_t)rem_rkeys
                                            + rem_rkeys_lengths[1] + rem_rkeys_lengths[0]),
                       &dpu_team->rem_data_out_key);
    free(rem_rkeys);

    *team = &dpu_team->super;
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_team_create_test(xccl_tl_team_t *team)
{
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_team_destroy(xccl_tl_team_t *team)
{
    xccl_dpu_team_t *dpu_team = ucs_derived_of(team, xccl_dpu_team_t);
    xccl_dpu_context_t *dpu_ctx = ucs_derived_of(team->ctx, xccl_dpu_context_t);
    dpu_sync_t hangup;
    ucx_request_t *hangup_req;
    ucp_request_param_t req_param;
 
    hangup.itt = dpu_team->coll_id;
    hangup.dtype = XCCL_DT_UNSUPPORTED;
    hangup.op = XCCL_OP_UNSUPPORTED;
    hangup.len = 0;
 
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_DATATYPE;
    req_param.datatype     = ucp_dt_make_contig(1);
    req_param.cb.send      = send_handler_nbx;
 
    hangup_req = ucp_put_nbx(dpu_ctx->ucp_ep, &hangup, sizeof(hangup),
                             dpu_team->rem_ctrl_seg, dpu_team->rem_ctrl_seg_key,
                             &req_param);
    if (ucx_req_check(hangup_req) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    do {
        ucp_worker_progress(dpu_ctx->ucp_worker);
    } while((ucx_req_test(&(hangup_req), dpu_ctx->ucp_worker) != XCCL_OK));
 
    ucp_rkey_destroy(dpu_team->rem_ctrl_seg_key);
    ucp_rkey_destroy(dpu_team->rem_data_in_key);
    ucp_rkey_destroy(dpu_team->rem_data_out_key);
    ucp_mem_unmap(dpu_ctx->ucp_context, dpu_team->ctrl_seg_memh);
    free(team);
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_init(xccl_coll_op_args_t *coll_args,
                                              xccl_tl_coll_req_t **request,
                                              xccl_tl_team_t *team)
{
    xccl_dpu_info("Collective init");
    xccl_dpu_coll_req_t *req = (xccl_dpu_coll_req_t*)malloc(sizeof(xccl_dpu_coll_req_t));
    xccl_dpu_team_t *dpu_team = ucs_derived_of(team, xccl_dpu_team_t);

    if (req == NULL) {
        return XCCL_ERR_NO_MEMORY;
    }
    memcpy(&req->args, coll_args, sizeof(xccl_coll_op_args_t));
    req->sync.itt = dpu_team->coll_id;
    req->sync.dtype = coll_args->reduce_info.dt;
    req->sync.len = coll_args->reduce_info.count;
    req->sync.op = coll_args->reduce_info.op;
    req->team = team;
    *request = &req->super;
    (*request)->lib = &xccl_team_lib_dpu.super;
    dpu_team->coll_id++;
    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_post(xccl_tl_coll_req_t *request)
{
    xccl_dpu_coll_req_t *req = ucs_derived_of(request, xccl_dpu_coll_req_t);
    xccl_dpu_team_t *dpu_team = ucs_derived_of(req->team, xccl_dpu_team_t);
    xccl_dpu_context_t *dpu_ctx = ucs_derived_of(req->team->ctx, xccl_dpu_context_t);
    ucp_request_param_t req_param;
    ucx_request_t *send_req[2];

    xccl_dpu_info("Collective post");
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_DATATYPE;
    req_param.datatype     = ucp_dt_make_contig(1);
    req_param.cb.send      = send_handler_nbx;

    send_req[0] = ucp_put_nbx(dpu_ctx->ucp_ep, req->args.buffer_info.src_buffer,
                              req->args.reduce_info.count * xccl_dt_size(req->args.reduce_info.dt),
                              dpu_team->rem_data_in, dpu_team->rem_data_in_key,
                              &req_param);
    if (ucx_req_check(send_req[0]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    ucp_worker_fence(dpu_ctx->ucp_worker);
    send_req[1] = ucp_put_nbx(dpu_ctx->ucp_ep, &req->sync, sizeof(req->sync),
                              dpu_team->rem_ctrl_seg, dpu_team->rem_ctrl_seg_key,
                              &req_param);
    if (ucx_req_check(send_req[1]) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    do {
        ucp_worker_progress(dpu_ctx->ucp_worker);
    } while((ucx_req_test(&(send_req[0]), dpu_ctx->ucp_worker) != XCCL_OK) ||
            (ucx_req_test(&(send_req[1]), dpu_ctx->ucp_worker) != XCCL_OK));

    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_wait(xccl_tl_coll_req_t *request)
{
    xccl_dpu_info("Collective wait");
    fprintf(stderr, "collective wait is not implemented");
    return XCCL_ERR_NOT_IMPLEMENTED;
}

static xccl_status_t xccl_dpu_collective_test(xccl_tl_coll_req_t *request)
{
    xccl_dpu_coll_req_t *req = ucs_derived_of(request, xccl_dpu_coll_req_t);
    xccl_dpu_team_t *dpu_team = ucs_derived_of(req->team, xccl_dpu_team_t);
    xccl_dpu_context_t *dpu_ctx = ucs_derived_of(req->team->ctx, xccl_dpu_context_t);
    ucp_request_param_t req_param;
    ucx_request_t *recv_req;
    volatile uint32_t *check_flag = dpu_team->ctrl_seg;

    if (dpu_team->coll_id != (*check_flag + 1)) {
        return XCCL_INPROGRESS;
    }
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_DATATYPE;
    req_param.datatype     = ucp_dt_make_contig(1);
    req_param.cb.recv      = recv_handler_nbx;

    recv_req = ucp_get_nbx(dpu_ctx->ucp_ep, req->args.buffer_info.dst_buffer,
                           req->args.reduce_info.count * xccl_dt_size(req->args.reduce_info.dt),
                           dpu_team->rem_data_out, dpu_team->rem_data_out_key,
                           &req_param);
    if (ucx_req_check(recv_req) != XCCL_OK) {
        return XCCL_ERR_NO_MESSAGE;
    }
    do {
        ucp_worker_progress(dpu_ctx->ucp_worker);
    } while((ucx_req_test(&recv_req, dpu_ctx->ucp_worker) != XCCL_OK));

    return XCCL_OK;
}

static xccl_status_t xccl_dpu_collective_finalize(xccl_tl_coll_req_t *request)
{
    xccl_dpu_info("Collective finalize");
    free(request);
    return XCCL_OK;
}

xccl_team_lib_dpu_t xccl_team_lib_dpu = {
    .super.name                   = "dpu",
    .super.id                     = XCCL_TL_DPU,
    .super.priority               = 90,
    .super.team_lib_config        =
    {
        .name                     = "DPU team library",
        .prefix                   = "TEAM_DPU_",
        .table                    = xccl_team_lib_dpu_config_table,
        .size                     = sizeof(xccl_team_lib_dpu_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "DPU tl context",
        .prefix                  = "TEAM_DPU_",
        .table                   = xccl_tl_dpu_context_config_table,
        .size                    = sizeof(xccl_tl_dpu_context_config_t),
    },
    .super.params.reproducible    = XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE,
    .super.params.thread_mode     = XCCL_THREAD_MODE_SINGLE |
                                    XCCL_THREAD_MODE_MULTIPLE,
    .super.params.team_usage      = XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES,
    .super.params.coll_types      = XCCL_COLL_CAP_ALLREDUCE,
    .super.mem_types              = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA),
    .super.ctx_create_mode        = XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL,
    .super.team_context_create    = xccl_dpu_context_create,
    .super.team_context_destroy   = xccl_dpu_context_destroy,
    .super.team_context_progress  = NULL,
    .super.team_create_post       = xccl_dpu_team_create_post,
    .super.team_create_test       = xccl_dpu_team_create_test,
    .super.team_destroy           = xccl_dpu_team_destroy,
    .super.team_lib_open          = xccl_dpu_lib_open,
    .super.collective_init        = xccl_dpu_collective_init,
    .super.collective_post        = xccl_dpu_collective_post,
    .super.collective_wait        = xccl_dpu_collective_wait,
    .super.collective_test        = xccl_dpu_collective_test,
    .super.collective_finalize    = xccl_dpu_collective_finalize,
    .super.global_mem_map_start   = NULL,
    .super.global_mem_map_test    = NULL,
    .super.global_mem_unmap       = NULL,
};
