/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_DEF_H_
#define XCCL_DEF_H_

#include <stddef.h>
#include <stdint.h>

typedef struct xccl_lib* xccl_lib_h;

/**
 * @ingroup XCCL_LIB_CONFIG
 * @brief XCCL configuration descriptor
 *
 * This descriptor defines the configuration for @ref xccl_lib_h
 * "XCCL team library". The configuration is loaded from the run-time
 * environment (using configuration files of environment variables)
 * using @ref xccl_lib_config_read "xccl_lib_config_read" routine and can be printed
 * using @ref xccl_lib_config_print "xccl_lib_config_print" routine. In addition,
 * application is responsible to release the descriptor using
 * @ref xccl_lib_config_release "xccl_lib_config_release" routine.
 */
typedef struct xccl_lib_config xccl_lib_config_t;

typedef struct xccl_context_config xccl_context_config_t;

/**
 * @ingroup XCCL_TEAM
 * @brief XCCL team handle
 *
 * @todo add description here
 */
typedef struct xccl_team* xccl_team_h;

/**
 * @ingroup XCCL_TEAM_CONTEXT
 * @brief XCCL team context handle
 *
 * @todo add description here
 */
typedef struct xccl_context* xccl_context_h;

/**
 * @ingroup XCCL_COLL
 * @brief XCCL collective requst handle
 *
 * @todo add description here
 */

typedef struct xccl_coll_req* xccl_coll_req_h;

typedef struct xccl_mem_handle* xccl_mem_h;

#endif
