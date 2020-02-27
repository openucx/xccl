/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef TCCL_H_
#define TCCL_H_

#include <api/tccl_def.h>
#include <api/tccl_version.h>
#include <api/tccl_status.h>

/**
 * @defgroup TCCL_API Unified Communication Collectives (TCCL) API
 * @{
 * This section describes TCCL API.
 * @}
 */
/**
 * @ingroup TCCL_TEAM_LIB
 * @brief @todo
 *
 */
typedef enum tccl_team_usage_type {
    TCCL_USAGE_HW_COLLECTIVES    = TCCL_BIT(0),
    TCCL_USAGE_SW_COLLECTIVES    = TCCL_BIT(1),
    TCCL_USAGE_P2P_NETWORK       = TCCL_BIT(2),
    TCCL_USAGE_HYBRID            = TCCL_BIT(3),
    TCCL_USAGE_NO_COMMUNICATION  = TCCL_BIT(4)
} tccl_team_usage_type_t;

typedef enum tccl_lib_config_field_mask {
    TCCL_LIB_CONFIG_FIELD_REPRODUCIBLE    = TCCL_BIT(0),
    TCCL_LIB_CONFIG_FIELD_THREAD_MODE     = TCCL_BIT(1),
    TCCL_LIB_CONFIG_FIELD_TEAM_USAGE      = TCCL_BIT(2),
    TCCL_LIB_CONFIG_FIELD_CONTEXT_CONFIG  = TCCL_BIT(3),
    TCCL_LIB_CONFIG_FIELD_TEAM_CONFIG     = TCCL_BIT(4),
    TCCL_LIB_CONFIG_FIELD_COLL_TYPES      = TCCL_BIT(5)
} tccl_lib_config_field_mask_t;

typedef enum tccl_reproducibility {
    TCCL_LIB_REPRODUCIBLE     = TCCL_BIT(0),
    TCCL_LIB_NON_REPRODUCIBLE = TCCL_BIT(1),
} tccl_reproducibility_t;

typedef enum tccl_thread_mode {
    TCCL_LIB_THREAD_MULTIPLE  = TCCL_BIT(0),
    TCCL_LIB_THREAD_SINGLE    = TCCL_BIT(1),
} tccl_thread_mode_t;

/**
 * @ingroup TCCL_TEAM_CONTEXT
 * @brief Completion symantics of collective operations
 *
 */
typedef enum {
    TCCL_TEAM_COMPLETION_BLOCKING    = 0,
    TCCL_TEAM_COMPLETION_NONBLOCKING = 1,
    TCCL_TEAM_COMPLETION_SPLIT_PHASE = 2
} tccl_team_completion_type_t;

typedef struct tccl_oob_collectives {
    int (*allgather)(void *src_buf, void *recv_buff, size_t size,
                     void *coll_context);
    void *coll_context;
    int rank;
    int size;
} tccl_oob_collectives_t;

typedef enum tccl_context_config_field_mask {
    TCCL_CONTEXT_CONFIG_FIELD_TEAM_LIB_NAME    = TCCL_BIT(0),
    TCCL_CONTEXT_CONFIG_FIELD_THREAD_MODE      = TCCL_BIT(1),
    TCCL_CONTEXT_CONFIG_FIELD_COMPLETION_TYPE  = TCCL_BIT(2),
    TCCL_CONTEXT_CONFIG_FIELD_OOB              = TCCL_BIT(3),
} tccl_context_config_field_mask_t;

typedef struct tccl_context_config {
    uint64_t                    field_mask;
    char*                       team_lib_name;
    tccl_thread_mode_t          thread_mode;
    tccl_team_completion_type_t completion_type;
    tccl_oob_collectives_t      oob;
} tccl_context_config_t;


typedef enum {
    TCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL  = 0,
    TCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL = 1
} tccl_team_lib_context_create_mode_t;

/**
 * @ingroup TCCL_TEAM_LIB
 * @brief TCCL team library attributes field mask.
 *
 * The enumeration allows specifying which fields in @ref tccl_team_lib_attr_t
 * are present. It is used to enable backward compatibility support.
 */
enum tccl_team_lib_attr_field {
    TCCL_ATTR_FIELD_CONTEXT_CREATE_MODE = 1
};

/**
 * @ingroup TCCL_TEAM_LIB
 * @brief Team library attributes.
 *
 * The structure defines the attributes which characterize
 * the particular team library.
 */
typedef struct tccl_team_lib_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref tccl_team_lib_attr_field.
     * Fields not specified in this mask will be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    uint64_t                           field_mask;

    /**
     * If not zero specifies that team context creation for the library
     * is global operation and requires oob context. Otherwise team context is
     * local operation and oob context may be omitted.
     */
    tccl_team_lib_context_create_mode_t context_create_mode;
} tccl_team_lib_attr_t;

/**
 * @ingroup TCCL_TEAM_LIB
 * @brief Get attributes specific to a particular team library.
 *
 * This routine fetches information about the team library.
 *
 * @param [in]  team_lib   Handle to @ref tccl_team_lib_h
 *                         "TCCL team library".
 *
 * @param [out] attr       Filled with attributes of @p team_lib library.
 *
 * @return Error code as defined by @ref tccl_status_t
 */
tccl_status_t tccl_team_lib_query(tccl_team_lib_h team_lib,
                                tccl_team_lib_attr_t *attr);

tccl_status_t tccl_create_context(tccl_lib_h lib,
                                  tccl_context_config_t config,
                                  tccl_context_h *team_ctx);

tccl_status_t tccl_destroy_context(tccl_context_h team_ctx);

/**
 * @ingroup TCCL_TEAM
 * @brief TCCL endpoint range descriptor type
 *
 * The enumeration specifies the type of endpoint range descriptor
 * passed to the @ref tccl_team_create_post as part of @ref tccl_team_config_t.
 *
 */
typedef enum {
    TCCL_EP_RANGE_UNDEFINED = 0, /**< The relation between the team EP and team_context rank is defined
                                   internally by the TCCL. User does not provide that info. */
    TCCL_EP_RANGE_STRIDED   = 1, /**< The ep range of the team can be described by the 2 values:
                                   start, stride. */
    TCCL_EP_RANGE_MAP       = 2, /**< The ep range is given as an array of intergers that map the ep in
                                 the team to the team_context rank. */
    TCCL_EP_RANGE_CB        = 3, /**< The ep range mapping is defined as callback provided by the TCCL user. */
} tccl_ep_range_type_t;

/**
 * @ingroup TCCL_TEAM
 * @brief TCCL Team endpoints range
 *
 * The structure defines how the range of endpoints that form a TCCL team is mapped to the
 * team context (i.e., a mapping to the world rank)
 * The number of endpoints is provided as part of @ref tccl_team_config_t.
 */
typedef struct tccl_ep_range_t {
    tccl_ep_range_type_t type;
    union {
        struct tccl_ep_range_strided {
            int start;
            int stride;
        } strided;
        struct tccl_ep_range_map {
            int *map;
        } map;
        struct tccl_ep_range_cb {
            int   (*cb)(int rank, void *cb_ctx);
            void  *cb_ctx;
        } cb;
    };
} tccl_ep_range_t;

typedef struct tccl_team_config {
    tccl_ep_range_t range;
} tccl_team_config_t;

tccl_status_t tccl_team_create_post(tccl_context_h team_ctx,
                                    tccl_team_config_h config,
                                    tccl_oob_collectives_t oob, tccl_team_h *team);

tccl_status_t tccl_team_destroy(tccl_team_h team);

typedef enum {
    TCCL_BARRIER     = TCCL_BIT(0),
    TCCL_ALLTOALL    = TCCL_BIT(1),
    TCCL_ALLTOALLV   = TCCL_BIT(2),
    TCCL_BCAST       = TCCL_BIT(3),
    TCCL_GATHER      = TCCL_BIT(4),
    TCCL_ALLGATHER   = TCCL_BIT(5),
    TCCL_REDUCE      = TCCL_BIT(6),
    TCCL_ALLREDUCE   = TCCL_BIT(7),
    TCCL_SCATTER     = TCCL_BIT(8),
    TCCL_FANIN       = TCCL_BIT(9),
    TCCL_FANOUT      = TCCL_BIT(10),
    TCCL_FLUSH_ALL   = TCCL_BIT(11),
    TCCL_MULTICAST   = TCCL_BIT(12),
    TCCL_COLL_ALL    = TCCL_MASK(13)
} tccl_collective_type_t;

typedef enum {
    TCCL_OP_MAX,
    TCCL_OP_MIN,
    TCCL_OP_SUM,
    TCCL_OP_PROD,
    TCCL_OP_AND,
    TCCL_OP_OR,
    TCCL_OP_XOR,
    TCCL_OP_LAND,
    TCCL_OP_LOR,
    TCCL_OP_LXOR,
    TCCL_OP_BAND,
    TCCL_OP_BOR,
    TCCL_OP_BXOR,
    TCCL_OP_MAXLOC,
    TCCL_OP_MINLOC,
    TCCL_OP_LAST_PREDEFINED,
    TCCL_OP_UNSUPPORTED
} tccl_op_t;

typedef enum {
    TCCL_DT_INT8,
    TCCL_DT_INT16,
    TCCL_DT_INT32,
    TCCL_DT_INT64,
    TCCL_DT_INT128,
    TCCL_DT_UINT8,
    TCCL_DT_UINT16,
    TCCL_DT_UINT32,
    TCCL_DT_UINT64,
    TCCL_DT_UINT128,
    TCCL_DT_FLOAT16,
    TCCL_DT_FLOAT32,
    TCCL_DT_FLOAT64,
    TCCL_DT_LAST_PREDEFINED,
    TCCL_DT_UNSUPPORTED
} tccl_dt_t;

typedef struct tccl_reduce_info {
    tccl_dt_t dt;
    tccl_op_t op;
    size_t   count;
} tccl_reduce_info_t;

typedef struct tccl_coll_buffer_info {
    void   *src_buffer;
    void   *dst_buffer;
    size_t len;
    int    flags;
} tccl_coll_buffer_info_t;

typedef struct tccl_coll_algorithm {
    uint16_t set_by_user : 1;
    uint16_t id : 15;
} tccl_coll_algorithm_t;

typedef struct tccl_coll_op_args {
    tccl_collective_type_t  coll_type;
    tccl_coll_buffer_info_t buffer_info;
    tccl_reduce_info_t      reduce_info;
    int                     root;
    tccl_coll_algorithm_t   alg;
    uint16_t                tag;
} tccl_coll_op_args_t;

typedef struct tccl_coll_req {
    tccl_team_lib_h lib;
} tccl_coll_req_t;

tccl_status_t tccl_collective_init(tccl_coll_op_args_t *coll_args,
                                 tccl_coll_req_h *request, tccl_team_h team);

tccl_status_t tccl_collective_post(tccl_coll_req_h request);

tccl_status_t tccl_collective_wait(tccl_coll_req_h request);

tccl_status_t tccl_collective_test(tccl_coll_req_h request);

tccl_status_t tccl_collective_finalize(tccl_coll_req_h request);

tccl_status_t tccl_context_progress(tccl_context_h context);

/**
 * @ingroup TCCL_LIB
 * @brief TCCL team library initializatoin parameters
 *
 */
typedef struct tccl_params {
    uint64_t                     field_mask;
    tccl_reproducibility_t       reproducible;
    tccl_thread_mode_t           thread_mode;
    tccl_team_usage_type_t       team_usage;
    tccl_collective_type_t       coll_types;
    tccl_context_config_t        context_config;
    tccl_team_config_t           team_config;
} tccl_params_t;

typedef struct tccl_config {
    char **tls;
} tccl_config_t;

/**
 * @ingroup UCP_TEAM_LIB
 * @brief Initialize team library
 *
 * @todo add description
 *
 * @param [in]  tccl_params (Library initialization parameters)
 * @param [out] team_lib   (TCCL team library handle)
 *
 * @return Error code
 */
tccl_status_t tccl_init(const tccl_params_t *params,
                        const tccl_config_t *config,
                        tccl_context_h *context_p);

tccl_status_t tccl_cleanup(tccl_context_h context_p);


static inline size_t tccl_dt_size(tccl_dt_t dt) {
    switch(dt) {
    case TCCL_DT_INT8:
    case TCCL_DT_UINT8:
        return 1;
    case TCCL_DT_INT16:
    case TCCL_DT_UINT16:
    case TCCL_DT_FLOAT16:
        return 2;
    case TCCL_DT_INT32:
    case TCCL_DT_UINT32:
    case TCCL_DT_FLOAT32:
        return 4;
    case TCCL_DT_INT64:
    case TCCL_DT_UINT64:
    case TCCL_DT_FLOAT64:
        return 8;
    case TCCL_DT_INT128:
    case TCCL_DT_UINT128:
        return 16;
    }
    return 0;
}
#endif
