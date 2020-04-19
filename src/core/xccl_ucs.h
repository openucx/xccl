/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef XCCL_UCS_H_
#define XCCL_UCS_H_

#include <inttypes.h>

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
