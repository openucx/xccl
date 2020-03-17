/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_DEF_H_
#define XCCL_DEF_H_
#include <stddef.h>
#include <stdint.h>

/**
 * @ingroup XCCL_TEAM_LIB
 * @brief XCCL library handle
 *
 * @todo add description here
 */
typedef struct xccl_team_lib* xccl_team_lib_h;
typedef struct xccl_lib* xccl_lib_h;

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
 * @ingroup XCCL_TEAM_CONTEXT
 * @brief XCCL team context config handle
 *
 * @todo add description here
 */
typedef struct xccl_context_config* xccl_context_config_h;

/**
 * @ingroup XCCL_TEAM
 * @brief XCCL team config handle
 *
 * @todo add description here
 */
typedef struct xccl_team_config* xccl_team_config_h;

/**
 * @ingroup XCCL_COLL
 * @brief XCCL collective requst handle
 *
 * @todo add description here
 */

typedef struct xccl_coll_req* xccl_coll_req_h;
typedef struct xccl_mem_handle* xccl_mem_h;

#define XCCL_BIT(i)               (1ul << (i))
#define XCCL_MASK(i)              (XCCL_BIT(i) - 1)
#define XCCL_PP_QUOTE(x)                 # x
#endif
