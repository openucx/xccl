/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef TEAM_UCX_TAG_H_
#define TEAM_UCX_TAG_H_
/*
 * UCX tag structure:
 *
 *    0123456          7                  01234567 01234567 01234567 01234567 01234567 01234567 01234567
 *                |             |                          |                          |
 *   RESERVED (7) | MCASTbit(1) |     message tag (16)     |     source rank (24)     |  context id (16)
 */
#define TEAM_UCX_RESERVED_BITS          7
#define TEAM_UCX_SCOLL_BITS             1
#define TEAM_UCX_TAG_BITS              16
#define TEAM_UCX_RANK_BITS             24
#define TEAM_UCX_CONTEXT_BITS          16

#define TEAM_UCX_RESERVED_BITS_OFFSET   (TEAM_UCX_CONTEXT_BITS + \
                                         TEAM_UCX_RANK_BITS + \
                                         TEAM_UCX_TAG_BITS + \
                                         TEAM_UCX_SCOLL_BITS)
#define TEAM_UCX_SCOLL_BITS_OFFSET   (TEAM_UCX_CONTEXT_BITS + \
                                      TEAM_UCX_RANK_BITS + TEAM_UCX_TAG_BITS)
#define TEAM_UCX_TAG_BITS_OFFSET     (TEAM_UCX_CONTEXT_BITS + TEAM_UCX_RANK_BITS)
#define TEAM_UCX_RANK_BITS_OFFSET    (TEAM_UCX_CONTEXT_BITS)
#define TEAM_UCX_CONTEXT_BITS_OFFSET 0

#define TEAM_UCX_MAX_CTAG    ((((uint64_t)1) << TEAM_UCX_CTAG_BITS) - 1)
#define TEAM_UCX_MAX_TAG     ((((uint64_t)1) << TEAM_UCX_TAG_BITS) - 1)
#define TEAM_UCX_MAX_RANK    ((((uint64_t)1) << TEAM_UCX_RANK_BITS) - 1)
#define TEAM_UCX_MAX_CONTEXT ((((uint64_t)1) << TEAM_UCX_CONTEXT_BITS) - 1)

#define TEAM_UCX_TAG_MASK        (TEAM_UCX_MAX_TAG     << TEAM_UCX_TAG_BITS_OFFSET)
#define TEAM_UCX_RANK_MASK       (TEAM_UCX_MAX_RANK    << TEAM_UCX_RANK_BITS_OFFSET)
#define TEAM_UCX_CONTEXT_MASK    (TEAM_UCX_MAX_CONTEXT << TEAM_UCX_CONTEXT_BITS_OFFSET)
#define TEAM_UCX_TAG_SENDER_MASK ((((uint64_t)1) << \
                                   (TEAM_UCX_CONTEXT_BITS + TEAM_UCX_RANK_BITS)) - 1)

#endif
