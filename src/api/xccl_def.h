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

static inline unsigned __ucs_ilog2_u32(uint32_t n)
{
    uint32_t result;
    __asm("bsrl %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

static inline unsigned __ucs_ilog2_u64(uint64_t n)
{
    uint64_t result;
    __asm("bsrq %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

static inline unsigned ucs_ffs64(uint64_t n)
{
    uint64_t result;
    __asm("bsfq %1,%0"
        : "=r" (result)
        : "r" (n));
    return result;
}

#define ucs_ilog2(_n)                   \
(                                       \
    __builtin_constant_p(_n) ? (        \
             (_n) < 1 ? 0 :             \
             (_n) & (1ULL << 17) ? 17 : \
             (_n) & (1ULL << 16) ? 16 : \
             (_n) & (1ULL << 15) ? 15 : \
             (_n) & (1ULL << 14) ? 14 : \
             (_n) & (1ULL << 13) ? 13 : \
             (_n) & (1ULL << 12) ? 12 : \
             (_n) & (1ULL << 11) ? 11 : \
             (_n) & (1ULL << 10) ? 10 : \
             (_n) & (1ULL <<  9) ?  9 : \
             (_n) & (1ULL <<  8) ?  8 : \
             (_n) & (1ULL <<  7) ?  7 : \
             (_n) & (1ULL <<  6) ?  6 : \
             (_n) & (1ULL <<  5) ?  5 : \
             (_n) & (1ULL <<  4) ?  4 : \
             (_n) & (1ULL <<  3) ?  3 : \
             (_n) & (1ULL <<  2) ?  2 : \
             (_n) & (1ULL <<  1) ?  1 : \
             (_n) & (1ULL <<  0) ?  0 : \
             0                          \
                                 ) :    \
    (sizeof(_n) <= 4) ?                 \
    __ucs_ilog2_u32((uint32_t)(_n)) :   \
    __ucs_ilog2_u64((uint64_t)(_n))     \
)
/* Returns the number of 1-bits in x */
#define ucs_popcount(_n) \
    ((sizeof(_n) <= 4) ? __builtin_popcount((uint32_t)(_n)) : __builtin_popcountl(_n))

#endif
