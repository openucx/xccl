/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_H_
#define XCCL_H_

#include <api/xccl_def.h>
#include <api/xccl_version.h>
#include <api/xccl_status.h>

/**
 * @defgroup XCCL_API Unified Communication Collectives (XCCL) API
 * @{
 * This section describes XCCL API.
 * @}
 */
/**
 * @ingroup XCCL_TEAM_LIB
 * @brief @todo
 *
 */
typedef enum xccl_team_usage_type {
    XCCL_USAGE_HW_COLLECTIVES    = XCCL_BIT(0),
    XCCL_USAGE_SW_COLLECTIVES    = XCCL_BIT(1),
    XCCL_USAGE_P2P_NETWORK       = XCCL_BIT(2),
    XCCL_USAGE_HYBRID            = XCCL_BIT(3),
    XCCL_USAGE_NO_COMMUNICATION  = XCCL_BIT(4)
} xccl_team_usage_type_t;

typedef enum xccl_lib_config_field_mask {
    XCCL_LIB_CONFIG_FIELD_REPRODUCIBLE    = XCCL_BIT(0),
    XCCL_LIB_CONFIG_FIELD_THREAD_MODE     = XCCL_BIT(1),
    XCCL_LIB_CONFIG_FIELD_TEAM_USAGE      = XCCL_BIT(2),
    XCCL_LIB_CONFIG_FIELD_CONTEXT_CONFIG  = XCCL_BIT(3),
    XCCL_LIB_CONFIG_FIELD_TEAM_CONFIG     = XCCL_BIT(4),
    XCCL_LIB_CONFIG_FIELD_COLL_TYPES      = XCCL_BIT(5)
} xccl_lib_config_field_mask_t;

typedef enum xccl_reproducibility {
    XCCL_LIB_REPRODUCIBLE     = XCCL_BIT(0),
    XCCL_LIB_NON_REPRODUCIBLE = XCCL_BIT(1),
} xccl_reproducibility_t;

typedef enum xccl_thread_mode {
    XCCL_LIB_THREAD_MULTIPLE  = XCCL_BIT(0),
    XCCL_LIB_THREAD_SINGLE    = XCCL_BIT(1),
} xccl_thread_mode_t;

/**
 * @ingroup XCCL_TEAM_CONTEXT
 * @brief Completion symantics of collective operations
 *
 */
typedef enum {
    XCCL_TEAM_COMPLETION_BLOCKING    = 0,
    XCCL_TEAM_COMPLETION_NONBLOCKING = 1,
    XCCL_TEAM_COMPLETION_SPLIT_PHASE = 2
} xccl_team_completion_type_t;

/**
 * @ingroup XCCL_TEAM
 * @brief XCCL endpoint range descriptor type
 *
 * The enumeration specifies the type of endpoint range descriptor
 * passed to the @ref xccl_team_create_post as part of @ref xccl_team_config_t.
 *
 */
typedef enum {
    XCCL_EP_RANGE_UNDEFINED = 0, /**< The relation between the team EP and team_context rank is defined
                                   internally by the XCCL. User does not provide that info. */
    XCCL_EP_RANGE_FULL      = 1, /**< The ep range of the team  spans all eps from a context*/
    XCCL_EP_RANGE_STRIDED   = 2, /**< The ep range of the team can be described by the 2 values:
                                   start, stride. */
    XCCL_EP_RANGE_MAP       = 3, /**< The ep range is given as an array of intergers that map the ep in
                                 the team to the team_context rank. */
    XCCL_EP_RANGE_CB        = 4, /**< The ep range mapping is defined as callback provided by the XCCL user. */
} xccl_ep_range_type_t;

/**
 * @ingroup XCCL_TEAM
 * @brief XCCL Team endpoints range
 *
 * The structure defines how the range of endpoints that form a XCCL team is mapped to the
 * team context (i.e., a mapping to the world rank)
 * The number of endpoints is provided as part of @ref xccl_team_config_t.
 */
typedef struct xccl_ep_range_t {
    xccl_ep_range_type_t type;
    int ep_num;
    union {
        struct xccl_ep_range_strided {
            int start;
            int stride;
        } strided;
        struct xccl_ep_range_map {
            int *map;
        } map;
        struct xccl_ep_range_cb {
            int   (*cb)(int rank, void *cb_ctx);
            void  *cb_ctx;
        } cb;
    };
} xccl_ep_range_t;

static inline int xccl_range_to_rank(xccl_ep_range_t range, int rank)
{
    int r;
    switch(range.type) {
    case XCCL_EP_RANGE_FULL:
        r = rank;
        break;
    case XCCL_EP_RANGE_STRIDED:
        r = range.strided.start + rank*range.strided.stride;
        break;
    case XCCL_EP_RANGE_MAP:
        r = range.map.map[rank];
        break;
    case XCCL_EP_RANGE_CB:
        r = range.cb.cb(rank, range.cb.cb_ctx);
        break;
    }
    return r;
}



typedef struct xccl_oob_collectives {
    int (*allgather)(void *src_buf, void *recv_buff, size_t size,
                     int my_rank, xccl_ep_range_t range,
                     void *coll_context);
    void *coll_context;
    int rank;
    int size;
} xccl_oob_collectives_t;

typedef enum xccl_context_config_field_mask {
    XCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE      = XCCL_BIT(0),
    XCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE  = XCCL_BIT(1),
    XCCL_CONTEXT_CONFIG_FIELD_OOB              = XCCL_BIT(2),
} xccl_context_config_field_mask_t;

typedef struct xccl_context_config {
    uint64_t                    field_mask;
    xccl_thread_mode_t          thread_mode;
    xccl_team_completion_type_t completion_type;
    xccl_oob_collectives_t      oob;
} xccl_context_config_t;


typedef enum {
    XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL  = 0,
    XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL = 1
} xccl_team_lib_context_create_mode_t;

/**
 * @ingroup XCCL_TEAM_LIB
 * @brief XCCL team library attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref xccl_team_lib_attr_t
 * are present. It is used to enable backward compatibility support.
 */
enum xccl_team_lib_attr_field {
    XCCL_ATTR_FIELD_CONTEXT_CREATE_MODE = 1
};

/**
 * @ingroup XCCL_TEAM_LIB
 * @brief Team library attributes.
 *
 * The structure defines the attributes which characterize
 * the particular team library.
 */
typedef struct xccl_team_lib_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref xccl_team_lib_attr_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                           field_mask;

    /**
     * If not zero specifies that team context creation for the library
     * is global operation and requires oob context. Otherwise team context is
     * local operation and oob context may be omitted.
     */
    xccl_team_lib_context_create_mode_t context_create_mode;
} xccl_team_lib_attr_t;

/**
 * @ingroup XCCL_TEAM_LIB
 * @brief Get attributes specific to a particular team library.
 *
 * This routine fetches information about the team library.
 *
 * @param [in]  team_lib   Handle to @ref xccl_team_lib_h
 *                         "XCCL team library".
 *
 * @param [out] attr       Filled with attributes of @p team_lib library.
 *
 * @return Error code as defined by @ref xccl_status_t
 */
xccl_status_t xccl_team_lib_query(xccl_team_lib_h team_lib,
                                xccl_team_lib_attr_t *attr);

typedef struct xccl_team_config {
    xccl_ep_range_t range;
} xccl_team_config_t;

xccl_status_t xccl_team_create_post(xccl_context_h team_ctx,
                                    xccl_team_config_h config,
                                    xccl_oob_collectives_t oob, xccl_team_h *team);

xccl_status_t xccl_team_destroy(xccl_team_h team);

typedef enum {
    XCCL_BARRIER = 0,
    XCCL_BCAST,
    XCCL_ALLREDUCE,
    XCCL_REDUCE,
    XCCL_FANIN,
    XCCL_FANOUT,
    XCCL_COLL_LAST
} xccl_collective_type_t;

typedef enum {
    XCCL_COLL_CAP_BARRIER     = XCCL_BIT(XCCL_BARRIER),
    XCCL_COLL_CAP_BCAST       = XCCL_BIT(XCCL_BCAST),
    XCCL_COLL_CAP_ALLREDUCE   = XCCL_BIT(XCCL_ALLREDUCE),
    XCCL_COLL_CAP_REDUCE      = XCCL_BIT(XCCL_REDUCE),
    XCCL_COLL_CAP_FANIN       = XCCL_BIT(XCCL_FANIN),
    XCCL_COLL_CAP_FANOUT      = XCCL_BIT(XCCL_FANOUT),
    XCCL_COLL_CAP_ALL         = XCCL_MASK(XCCL_COLL_LAST)
} xccl_collective_cap_t;

typedef enum {
    XCCL_OP_MAX,
    XCCL_OP_MIN,
    XCCL_OP_SUM,
    XCCL_OP_PROD,
    XCCL_OP_AND,
    XCCL_OP_OR,
    XCCL_OP_XOR,
    XCCL_OP_LAND,
    XCCL_OP_LOR,
    XCCL_OP_LXOR,
    XCCL_OP_BAND,
    XCCL_OP_BOR,
    XCCL_OP_BXOR,
    XCCL_OP_MAXLOC,
    XCCL_OP_MINLOC,
    XCCL_OP_LAST_PREDEFINED,
    XCCL_OP_UNSUPPORTED
} xccl_op_t;

typedef enum {
    XCCL_DT_INT8,
    XCCL_DT_INT16,
    XCCL_DT_INT32,
    XCCL_DT_INT64,
    XCCL_DT_INT128,
    XCCL_DT_UINT8,
    XCCL_DT_UINT16,
    XCCL_DT_UINT32,
    XCCL_DT_UINT64,
    XCCL_DT_UINT128,
    XCCL_DT_FLOAT16,
    XCCL_DT_FLOAT32,
    XCCL_DT_FLOAT64,
    XCCL_DT_LAST_PREDEFINED,
    XCCL_DT_UNSUPPORTED
} xccl_dt_t;

typedef struct xccl_reduce_info {
    xccl_dt_t dt;
    xccl_op_t op;
    size_t   count;
} xccl_reduce_info_t;

typedef struct xccl_coll_buffer_info {
    void   *src_buffer;
    void   *dst_buffer;
    size_t len;
    int    flags;
} xccl_coll_buffer_info_t;

typedef struct xccl_coll_algorithm {
    uint16_t set_by_user : 1;
    uint16_t id : 15;
} xccl_coll_algorithm_t;

typedef struct xccl_coll_op_args {
    xccl_collective_type_t  coll_type;
    xccl_coll_buffer_info_t buffer_info;
    xccl_reduce_info_t      reduce_info;
    int                     root;
    xccl_coll_algorithm_t   alg;
    uint16_t                tag;
} xccl_coll_op_args_t;

typedef struct xccl_coll_req {
    xccl_team_lib_h lib;
} xccl_coll_req_t;

xccl_status_t xccl_collective_init(xccl_coll_op_args_t *coll_args,
                                 xccl_coll_req_h *request, xccl_team_h team);

xccl_status_t xccl_collective_post(xccl_coll_req_h request);

xccl_status_t xccl_collective_wait(xccl_coll_req_h request);

xccl_status_t xccl_collective_test(xccl_coll_req_h request);

xccl_status_t xccl_collective_finalize(xccl_coll_req_h request);

xccl_status_t xccl_context_progress(xccl_context_h context);

/**
 * @ingroup XCCL_LIB
 * @brief XCCL team library initializatoin parameters
 *
 */
typedef struct xccl_params {
    uint64_t                     field_mask;
    xccl_reproducibility_t       reproducible;
    xccl_thread_mode_t           thread_mode;
    xccl_team_usage_type_t       team_usage;
    xccl_collective_cap_t        coll_types;
    xccl_context_config_t        context_config;
    xccl_team_config_t           team_config;
} xccl_params_t;

typedef struct xccl_config {
    xccl_context_config_t ctx_config;
    const char *tls;
} xccl_config_t;

/**
 * @ingroup UCP_TEAM_LIB
 * @brief Initialize team library
 *
 * @todo add description
 *
 * @param [in]  xccl_params (Library initialization parameters)
 * @param [out] team_lib   (XCCL team library handle)
 *
 * @return Error code
 */
xccl_status_t xccl_init(const xccl_params_t *params,
                        const xccl_config_t *config,
                        xccl_context_h *context_p);

xccl_status_t xccl_cleanup(xccl_context_h context_p);


static inline size_t xccl_dt_size(xccl_dt_t dt) {
    switch(dt) {
    case XCCL_DT_INT8:
    case XCCL_DT_UINT8:
        return 1;
    case XCCL_DT_INT16:
    case XCCL_DT_UINT16:
    case XCCL_DT_FLOAT16:
        return 2;
    case XCCL_DT_INT32:
    case XCCL_DT_UINT32:
    case XCCL_DT_FLOAT32:
        return 4;
    case XCCL_DT_INT64:
    case XCCL_DT_UINT64:
    case XCCL_DT_FLOAT64:
        return 8;
    case XCCL_DT_INT128:
    case XCCL_DT_UINT128:
        return 16;
    }
    return 0;
}
#endif
