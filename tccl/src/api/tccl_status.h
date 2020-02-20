/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TCCL_STATUS_H_
#define TCCL_STATUS_H_

typedef enum {
    /* Operation completed successfully */
    TCCL_OK                         =   0,

    /* Operation is queued and still in progress */
    TCCL_INPROGRESS                 =   1,

    /* Failure codes */
    TCCL_ERR_NO_MESSAGE             =  -1,
    TCCL_ERR_NO_RESOURCE            =  -2,
    TCCL_ERR_NO_MEMORY              =  -4,
    TCCL_ERR_INVALID_PARAM          =  -5,
    TCCL_ERR_UNREACHABLE            =  -6,    
    TCCL_ERR_NOT_IMPLEMENTED        =  -8,
    TCCL_ERR_MESSAGE_TRUNCATED      =  -9,
    TCCL_ERR_NO_PROGRESS            = -10,
    TCCL_ERR_BUFFER_TOO_SMALL       = -11,
    TCCL_ERR_NO_ELEM                = -12,
    TCCL_ERR_UNSUPPORTED            = -22,
    TCCL_ERR_LAST                   = -100
} tccl_status_t;
#endif
