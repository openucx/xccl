/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "xccl_mhba_lib.h"
//todo check not-needed includes
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <sys/un.h>

typedef struct
{
    int sock, fd;
    uint32_t pd_handle;
    xccl_status_t return_val;
} connection_t;

void* do_sendmsg(void *ptr)
{
    connection_t* conn = (connection_t *)ptr;

    struct msghdr msg = {};
    struct cmsghdr *cmsghdr;
    struct iovec iov[1];
    ssize_t nbytes;
    int *p;
    char buf[CMSG_SPACE(sizeof(int))];
    uint32_t handles[1];

    handles[0] = conn->pd_handle;

    iov[0].iov_base = handles;
    iov[0].iov_len = sizeof(handles);
    memset(buf, 0x0b, sizeof(buf));
    cmsghdr = (struct cmsghdr *)buf;
    cmsghdr->cmsg_len = CMSG_LEN(sizeof(int));
    cmsghdr->cmsg_level = SOL_SOCKET;
    cmsghdr->cmsg_type = SCM_RIGHTS;
    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_iov = iov;
    msg.msg_iovlen = sizeof(iov) / sizeof(iov[0]);
    msg.msg_control = cmsghdr;
    msg.msg_controllen = CMSG_LEN(sizeof(int));
    msg.msg_flags = 0;
    p = (int *)CMSG_DATA(cmsghdr);
    *p = conn->fd;
    xccl_mhba_debug("sendmsg: %d, msg.msg_iovlen=%d\n", conn->fd, (int)msg.msg_iovlen);

    nbytes = sendmsg(conn->sock, &msg, 0);
    if (nbytes == -1){
        xccl_mhba_error("Fail to send msg");
        conn->return_val = XCCL_ERR_NO_MESSAGE;
    }

    xccl_mhba_debug("sendmsg: nbytes=%u\n", (int)nbytes);
    conn->return_val = XCCL_OK;
}

static xccl_status_t do_recvmsg(int sock, int* shared_cmd_fd, uint32_t* shared_pd_handle)
{
    struct msghdr msg;
    struct cmsghdr *cmsghdr;
    struct iovec iov[1];
    ssize_t nbytes;
    int *p;
    char buf[CMSG_SPACE(sizeof(int))];
    uint32_t handles[1] = {};

    iov[0].iov_base = handles;
    iov[0].iov_len = sizeof(handles);

    xccl_mhba_debug("do_recvmsg: got len of %ld\n", iov[0].iov_len);
    memset(buf, 0x0d, sizeof(buf));
    cmsghdr = (struct cmsghdr *)buf;
    cmsghdr->cmsg_len = CMSG_LEN(sizeof(int));
    cmsghdr->cmsg_level = SOL_SOCKET;
    cmsghdr->cmsg_type = SCM_RIGHTS;
    msg.msg_name = NULL;
    msg.msg_namelen = 0;
    msg.msg_iov = iov;
    msg.msg_iovlen = sizeof(iov) / sizeof(iov[0]);
    msg.msg_control = cmsghdr;
    msg.msg_controllen = CMSG_LEN(sizeof(int));
    msg.msg_flags = 0;

    nbytes = recvmsg(sock, &msg, 0);
    if (nbytes == -1){
        xccl_mhba_error("Failed to recv msg");
        return XCCL_ERR_NO_MESSAGE;
    }

    p = (int *)CMSG_DATA(cmsghdr);
    xccl_mhba_debug("recvmsg: %d, nbytes=%d, handle1=%d\n", *p, (int)nbytes, handles[0]);

    *shared_cmd_fd = *p;
    *shared_pd_handle = handles[0];

    return XCCL_OK;
}

static xccl_status_t client_recv_data(int* shared_cmd_fd,uint32_t *shared_pd_handle, char *sock_path)
{
    int sock;
    struct sockaddr_storage sockaddr = {};
    struct sockaddr_un *addr;

    sock = socket(PF_LOCAL, SOCK_STREAM, 0);
    if (sock == -1)
    {
        xccl_mhba_error("Failed to create client socket errno %d", errno);
        return XCCL_ERR_NO_MESSAGE;
    }

    addr = (struct sockaddr_un *)&sockaddr;
    addr->sun_family = AF_UNIX;
    strncpy(addr->sun_path, sock_path, sizeof(addr->sun_path));
    addr->sun_path[sizeof(addr->sun_path) -1 ] = '\0';

    if (connect(sock, (struct sockaddr *)addr, SUN_LEN(addr)) == -1) {
        xccl_mhba_error("Failed to connect client socket errno %d", errno);
        goto fail;
    }
    if (do_recvmsg(sock, shared_cmd_fd, shared_pd_handle) != XCCL_OK) {
        xccl_mhba_error("Failed to recv msg");
        goto fail;
    }

    if (close(sock) == -1){
        xccl_mhba_error("Failed to close client socket errno %d", errno);
        return XCCL_ERR_NO_MESSAGE;
    }
    return XCCL_OK;

fail:
    if (close(sock) == -1){
        xccl_mhba_error("Failed to close client socket errno %d", errno);
    }
    return XCCL_ERR_NO_MESSAGE;
}

static xccl_status_t server_send_data(int command_fd, uint32_t pd_handle, int num_of_connections, xccl_mhba_node_t
                                        *node, xccl_team_params_t *params, char *sock_path)
{
    int sock, i, num_of_curr_connections = 0;
    struct sockaddr_storage storage= {};
    struct sockaddr_un *addr;

    connection_t connection[num_of_connections];
    pthread_t thread[num_of_connections];

    sock = socket(PF_LOCAL, SOCK_STREAM, 0);
    if (sock == -1){
        xccl_mhba_error("Failed to create server socket errno %d", errno);
        return XCCL_ERR_NO_MESSAGE;
    }

    addr = (struct sockaddr_un *)&storage;
    addr->sun_family = AF_UNIX;
    strncpy(addr->sun_path, sock_path, sizeof(addr->sun_path));
    addr->sun_path[sizeof(addr->sun_path) -1 ] = '\0';

    if (bind(sock, (struct sockaddr *)addr, sizeof(struct sockaddr_un)) == -1) {
        xccl_mhba_error("Failed to bind server socket errno %d", errno);
        goto fail;
    }
    if (listen(sock, num_of_connections) == -1) {
        xccl_mhba_error("Failed to listen to server socket errno %d", errno);
        goto listen_fail;
    }

    xccl_sbgp_oob_barrier(node->sbgp, params->oob);

    while (num_of_curr_connections < num_of_connections)
    {
        /* accept incoming connections */
        connection[num_of_curr_connections].sock = accept(sock, NULL, 0);
        connection[num_of_curr_connections].fd = command_fd;
        connection[num_of_curr_connections].pd_handle = pd_handle;
        if (connection[num_of_curr_connections].sock != -1)
        {
            /* start a new thread but do not wait for it */
            pthread_create(&thread[num_of_curr_connections], 0, do_sendmsg, (void *)
            &connection[num_of_curr_connections]);
            num_of_curr_connections += 1;
        }
    }

    for(i=0; i<num_of_connections;i++){
        pthread_join(thread[i],NULL);
    }
    for(i=0; i<num_of_connections;i++){
        if (connection[i].return_val != XCCL_OK){
            xccl_mhba_error("Failed to send cmd_fd");
            goto listen_fail;
        }
    }

    if (close(sock) == -1){
        xccl_mhba_error("Failed to close server socket errno %d", errno);
        if(remove(addr->sun_path)==-1){
            xccl_mhba_error("Socket file removal failed");
        }
        return XCCL_ERR_NO_MESSAGE;
    }

    if(remove(addr->sun_path)==-1){
        xccl_mhba_error("Socket file removal failed");
        return XCCL_ERR_NO_MESSAGE;
    }

    return XCCL_OK;

listen_fail:
    if(remove(addr->sun_path)==-1){
        xccl_mhba_error("Socket file removal failed");
    }
fail:
    if (close(sock) == -1){
        xccl_mhba_error("Failed to close server socket errno %d", errno);
    }
    return XCCL_ERR_NO_MESSAGE;
}

xccl_status_t xccl_mhba_share_ctx_pd(int root, xccl_mhba_node_t *node, int ctx_fd, uint32_t pd_handle,
                                     xccl_mhba_context_t *ctx, xccl_team_params_t *params, char *sock_path){
    int      shared_ctx_fd;
    uint32_t shared_pd_handle;
    xccl_status_t status;
    if (root != node->sbgp->group_rank) {
        xccl_sbgp_oob_barrier(node->sbgp, params->oob);
        status = client_recv_data(&shared_ctx_fd,&shared_pd_handle, sock_path);
        if (XCCL_OK != status){
            return status;
        }
        node->shared_ctx = ibv_import_device(shared_ctx_fd);
        if (!node->shared_ctx) {
            xccl_mhba_error("Import context failed");
            return XCCL_ERR_NO_MESSAGE;
        }
        node->shared_pd = ibv_import_pd(node->shared_ctx, shared_pd_handle); //todo if working - allocate pd from shared ctx instead of importing - should work
        if (!node->shared_pd) {
            xccl_mhba_error("Import PD failed");
            if(ibv_close_device(node->shared_ctx)){
                xccl_mhba_error("imported context close failed");
            }
            return XCCL_ERR_NO_MESSAGE;
        }
    } else{
        status = server_send_data(ctx_fd,pd_handle,node->sbgp->group_size -1, node, params, sock_path);
        if (XCCL_OK != status){
            xccl_mhba_error("Failed to Share ctx & pd from server side");
            return status;
        }
        node->shared_ctx = ctx->ib_ctx;
        node->shared_pd  = ctx->ib_pd;
    }
    return XCCL_OK;
}

xccl_status_t xccl_mhba_remove_shared_ctx_pd(int root, xccl_mhba_node_t *node){
    if (root != node->sbgp->group_rank) {
        ibv_unimport_pd(node->shared_pd);
        if(ibv_close_device(node->shared_ctx)){
            xccl_mhba_error("imported context close failed");
            return XCCL_ERR_NO_MESSAGE;
        }
    }
    return XCCL_OK;
}
