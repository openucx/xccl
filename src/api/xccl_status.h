/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_STATUS_H_
#define XCCL_STATUS_H_

typedef enum {
    /* Operation completed successfully */
    XCCL_OK                         =   0,

    /* Operation is queued and still in progress */
    XCCL_INPROGRESS                 =   1,

    /* Operation is queued but has not started yet*/
    XCCL_INITIALIZED                =   2,

    /* Failure codes */
    XCCL_ERR_NO_MESSAGE             =  -1,
    XCCL_ERR_NO_RESOURCE            =  -2,
    XCCL_ERR_NO_MEMORY              =  -4,
    XCCL_ERR_INVALID_PARAM          =  -5,
    XCCL_ERR_UNREACHABLE            =  -6,
    XCCL_ERR_NOT_IMPLEMENTED        =  -8,
    XCCL_ERR_MESSAGE_TRUNCATED      =  -9,
    XCCL_ERR_NO_PROGRESS            = -10,
    XCCL_ERR_BUFFER_TOO_SMALL       = -11,
    XCCL_ERR_NO_ELEM                = -12,
    XCCL_ERR_UNSUPPORTED            = -22,
    XCCL_ERR_LAST                   = -100
} xccl_status_t;
#endif
