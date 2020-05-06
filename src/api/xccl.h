/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef XCCL_H_
#define XCCL_H_

#include <api/xccl_def.h>
#include <api/xccl_version.h>
#include <api/xccl_tls.h>
#include <api/xccl_status.h>
#include <ucs/config/types.h>
#include <stdio.h>

BEGIN_C_DECLS

/**
 * @defgroup XCCL_API Unified Communication Collectives (XCCL) API
 * @{
 * This section describes XCCL API.
 * @}
 */
/**
 * @ingroup XCCL_LIB
 * @brief @todo
 *
 */

enum xccl_lib_params_field {
    XCCL_LIB_PARAM_FIELD_REPRODUCIBLE = UCS_BIT(0),
    XCCL_LIB_PARAM_FIELD_THREAD_MODE  = UCS_BIT(1),
    XCCL_LIB_PARAM_FIELD_TEAM_USAGE   = UCS_BIT(2),
    XCCL_LIB_PARAM_FIELD_COLL_TYPES   = UCS_BIT(4)
};

enum xccl_reproducibility_mode {
    XCCL_REPRODUCIBILITY_MODE_REPRODUCIBLE     = UCS_BIT(0),
    XCCL_REPRODUCIBILITY_MODE_NON_REPRODUCIBLE = UCS_BIT(1),
};

enum xccl_thread_mode {
    XCCL_THREAD_MODE_MULTIPLE = UCS_BIT(0),
    XCCL_THREAD_MODE_SINGLE   = UCS_BIT(1),
};

enum xccl_lib_params_team_usage_field {
    XCCL_LIB_PARAMS_TEAM_USAGE_HW_COLLECTIVES   = UCS_BIT(0),
    XCCL_LIB_PARAMS_TEAM_USAGE_SW_COLLECTIVES   = UCS_BIT(1),
};

typedef enum {
    XCCL_BARRIER = 0,
    XCCL_BCAST,
    XCCL_ALLREDUCE,
    XCCL_REDUCE,
    XCCL_ALLTOALL,
    XCCL_FANIN,
    XCCL_FANOUT,
    XCCL_FANOUT_GET,
    XCCL_COLL_LAST
} xccl_collective_type_t;

typedef enum {
    XCCL_COLL_CAP_BARRIER    = UCS_BIT(XCCL_BARRIER),
    XCCL_COLL_CAP_BCAST      = UCS_BIT(XCCL_BCAST),
    XCCL_COLL_CAP_ALLREDUCE  = UCS_BIT(XCCL_ALLREDUCE),
    XCCL_COLL_CAP_REDUCE     = UCS_BIT(XCCL_REDUCE),
    XCCL_COLL_CAP_ALLTOALL   = UCS_BIT(XCCL_ALLTOALL),
    XCCL_COLL_CAP_FANIN      = UCS_BIT(XCCL_FANIN),
    XCCL_COLL_CAP_FANOUT     = UCS_BIT(XCCL_FANOUT),
    XCCL_COLL_CAP_FANOUT_GET = UCS_BIT(XCCL_FANOUT_GET),
    XCCL_COLL_CAP_ALL        = UCS_MASK(XCCL_COLL_LAST)
} xccl_collective_cap_t;

/**
 * @ingroup XCCL_LIB
 * @brief XCCL library initializatoin parameters
 */
typedef struct xccl_lib_params {
    uint64_t field_mask;
    unsigned reproducible;
    unsigned thread_mode;
    uint64_t team_usage;
    uint64_t coll_types;
} xccl_lib_params_t;

/**
 * @ingroup XCCL_CONFIG
 * @brief Read XCCL configuration descriptor
 *
 * The routine fetches the information about XCCL configuration from
 * the run-time environment. Then, the fetched descriptor is used for
 * XCCL @ref xccl_lib_init "initialization". In addition
 * the application is responsible for @ref xccl_lib_config_release "releasing"
 * the descriptor back to the XCCL.
 *
 * @param [in]  env_prefix    If non-NULL, the routine searches for the
 *                            environment variables that start with
 *                            @e \<env_prefix\>_XCCL_ prefix.
 *                            Otherwise, the routine searches for the
 *                            environment variables that start with
 *                            @e XCCL_ prefix.
 * @param [in]  filename      If non-NULL, read configuration from the file
 *                            defined by @e filename. If the file does not
 *                            exist, it will be ignored and no error reported
 *                            to the application.
 * @param [out] config_p      Pointer to configuration descriptor as defined by
 *                            @ref xccl_lib_config_t "xccl_lib_config_t".
 *
 * @return Error code as defined by @ref xccl_status_t
 */

xccl_status_t xccl_lib_config_read(const char *env_prefix, const char *filename,
                                   xccl_lib_config_t **config_p);

/**
 * @ingroup XCCL_CONFIG
 * @brief Release configuration descriptor
 *
 * The routine releases the configuration descriptor that was allocated through
 * @ref xccl_config_read "xccl_config_read()" routine.
 *
 * @param [in] config        Configuration descriptor as defined by
 *                            @ref xccl_lib_config_t "xccl_lib_config_t".
 */

void xccl_lib_config_release(xccl_lib_config_t *config);

/**
 * @ingroup XCCL_CONFIG
 * @brief Print configuration information
 *
 * The routine prints the configuration information that is stored in
 * @ref xccl_lib_config_t "configuration" descriptor.
 *
 * @param [in]  config        @ref xccl_lib_config_t "Configuration descriptor"
 *                            to print.
 * @param [in]  stream        Output stream to print the configuration to.
 * @param [in]  title         Configuration title to print.
 * @param [in]  print_flags   Flags that control various printing options.
 */
void xccl_lib_config_print(const xccl_lib_config_t *config, FILE *stream,
                           const char *title,
                           ucs_config_print_flags_t print_flags);

/**
 * @ingroup XCCL_LIB
 * @brief Initialize XCCL library.
 *
 * @todo add description
 *
 * @param [in]  params    (Library initialization parameters)
 * @param [in]  config    XCCL configuration descriptor allocated through
 *                        @ref xccl_config_read "xccl_config_read()" routine.
 * @param [out] lib       (XCCL library handle)
 *
 * @return Error code
 */
xccl_status_t xccl_lib_init(const xccl_lib_params_t *params,
                            const xccl_lib_config_t *config,
                            xccl_lib_h *lib_p);

/**
 * @ingroup XCCL_LIB
 * @brief Release XCCL library.
 *
 * @todo add description
 *
 * @param [in] lib_p   Handle to @ref xccl_lib_h
 *                     "XCCL library".
 */
void xccl_lib_cleanup(xccl_lib_h lib_p);


xccl_status_t xccl_get_tl_list(xccl_lib_h lib, xccl_tl_id_t **tls,
                               unsigned *tl_count);

void xccl_free_tl_list(xccl_tl_id_t *tls);

enum xccl_tl_attr_field {
    XCCL_TL_ATTR_FIELD_CONTEXT_CREATE_MODE = UCS_BIT(0),
    XCCL_TL_ATTR_FIELD_DEVICES_COUNT       = UCS_BIT(1),
    XCCL_TL_ATTR_FILED_DEVICES             = UCS_BIT(2)
};

typedef enum {
    XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_LOCAL  = 0,
    XCCL_TEAM_LIB_CONTEXT_CREATE_MODE_GLOBAL = 1
} xccl_context_create_mode_t;

typedef struct xccl_tl_attr {
    uint64_t                   field_mask;
    xccl_context_create_mode_t context_create_mode;
    int                        devices_count;
    char                       (*devices)[16];
} xccl_tl_attr_t;

xccl_status_t xccl_tl_query(xccl_lib_h lib, xccl_tl_id_t *tl,
                            xccl_tl_attr_t *tl_attr);
void xccl_free_tl_attr(xccl_tl_attr_t *attr);

enum xccl_ctx_attr_field {
    XCCL_CTX_ATTR_FIELD_SUPPORTED_COLLS = UCS_BIT(0),
};

typedef struct xccl_ctx_attr {
    uint64_t  field_mask;
    uint64_t  supported_colls;
} xccl_ctx_attr_t;

xccl_status_t xccl_ctx_query(xccl_context_h ctx, xccl_ctx_attr_t *attr);
void xccl_free_ctx_attr(xccl_ctx_attr_t *attr);

enum xccl_context_params_field {
    XCCL_CONTEXT_PARAM_FIELD_THREAD_MODE          = UCS_BIT(0),
    XCCL_CONTEXT_PARAM_FIELD_TEAM_COMPLETION_TYPE = UCS_BIT(1),
    XCCL_CONTEXT_PARAM_FIELD_OOB                  = UCS_BIT(2),
    XCCL_CONTEXT_PARAM_FIELD_TLS                  = UCS_BIT(3)
};

/**
 * @ingroup XCCL_TEAM_CONTEXT
 * @brief Completion symantics of collective operations
 *
 */
typedef enum xccl_team_completion_type{
    XCCL_TEAM_COMPLETION_TYPE_BLOCKING    = 0,
    XCCL_TEAM_COMPLETION_TYPE_NONBLOCKING = 1,
    XCCL_TEAM_COMPLETION_TYPE_SPLIT_PHASE = 2
} xccl_team_completion_type_t;

/**
 * @ingroup XCCL_TEAM
 * @brief XCCL endpoint range descriptor type
 *
 * The enumeration specifies the type of endpoint range descriptor
 * passed to the @ref xccl_team_create_post as part of @ref xccl_team_params_t.
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
 * The number of endpoints is provided as part of @ref xccl_team_params_t.
 */
struct xccl_ep_range_strided {
    int start;
    int stride;
};

struct xccl_ep_range_map {
    int *map;
};

struct xccl_ep_range_cb {
    int   (*cb)(int rank, void *cb_ctx);
    void  *cb_ctx;
};

typedef struct xccl_ep_range_t {
    xccl_ep_range_type_t type;
    int                  ep_num;
    union {
        struct xccl_ep_range_strided strided;
        struct xccl_ep_range_map     map;
        struct xccl_ep_range_cb      cb;
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
    int           (*allgather)(void *src_buf, void *recv_buff, size_t size,
                               int my_rank, xccl_ep_range_t range,
                               void *coll_context, void **request);
    xccl_status_t (*req_test)(void *request);
    xccl_status_t (*req_free)(void *request);
    void          *coll_context;
    int           rank;
    int           size;
} xccl_oob_collectives_t;


typedef struct xccl_context_params {
    uint64_t                    field_mask;
    unsigned                    thread_mode;
    xccl_team_completion_type_t completion_type;
    xccl_oob_collectives_t      oob;
    uint64_t                    tls;
} xccl_context_params_t;

xccl_status_t xccl_context_config_read(xccl_lib_h lib, const char *env_prefix,
                                       const char *filename,
                                       xccl_context_config_t **config);

xccl_status_t xccl_context_config_modify(xccl_tl_id_t *tl_id,
                                         xccl_context_config_t *config,
                                         const char *name, const char *value);

void xccl_context_config_release(xccl_context_config_t *config);


xccl_status_t xccl_context_create(xccl_lib_h lib,
                                  const xccl_context_params_t *params,
                                  const xccl_context_config_t *config,
                                  xccl_context_h *context);

xccl_status_t xccl_context_progress(xccl_context_h context);

void xccl_context_destroy(xccl_context_h context);

enum xccl_team_params_field {
    XCCL_TEAM_PARAM_FIELD_EP_RANGE = UCS_BIT(0),
    XCCL_TEAM_PARAM_FIELD_OOB      = UCS_BIT(1)
};

typedef struct xccl_team_params {
    uint64_t               field_mask;
    xccl_ep_range_t        range;
    xccl_oob_collectives_t oob;
} xccl_team_params_t;

xccl_status_t xccl_team_create_post(xccl_context_h context,
                                    xccl_team_params_t *params,
                                    xccl_team_h *team);

xccl_status_t xccl_team_create_test(xccl_team_h team);

void xccl_team_destroy(xccl_team_h team);

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
    size_t    count;
} xccl_reduce_info_t;

typedef struct xccl_coll_buffer_info {
    void   *src_buffer;
    void   *dst_buffer;
    size_t len;
} xccl_coll_buffer_info_t;

typedef struct xccl_coll_get_info {
    xccl_mem_h memh;
    ptrdiff_t  offset;
    void       *local_buffer;
    size_t len;
} xccl_coll_get_info_t;

typedef struct xccl_coll_algorithm {
    uint8_t set_by_user : 1;
    uint8_t id : 7;
} xccl_coll_algorithm_t;


typedef struct xccl_coll_op_args {
    xccl_collective_type_t  coll_type;
    union {
        xccl_coll_buffer_info_t buffer_info;
        xccl_coll_get_info_t    get_info;
    };
    xccl_reduce_info_t      reduce_info;
    int                     root;
    xccl_coll_algorithm_t   alg;
    uint16_t                tag;
} xccl_coll_op_args_t;

xccl_status_t xccl_collective_init(xccl_coll_op_args_t *coll_args,
                                   xccl_coll_req_h *request, xccl_team_h team);

xccl_status_t xccl_collective_post(xccl_coll_req_h request);

xccl_status_t xccl_collective_wait(xccl_coll_req_h request);

xccl_status_t xccl_collective_test(xccl_coll_req_h request);

xccl_status_t xccl_collective_finalize(xccl_coll_req_h request);


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

typedef enum xccl_mem_map_params_field_mask {
    XCCL_MEM_MAP_PARAM_FIELD_ADDRESS = UCS_BIT(0),
    XCCL_MEM_MAP_PARAM_FIELD_LENGTH  = UCS_BIT(1),
    XCCL_MEM_MAP_PARAM_FIELD_ROOT    = UCS_BIT(2)
} xccl_mem_map_params_field_mask_t;

typedef struct xccl_mem_map_params {
    uint64_t field_mask;
    /**
     * If the address is not NULL, the routine maps (registers) the memory segment
     * pointed to by this address.
     * If the pointer is NULL, the library allocates mapped (registered) memory
     * segment.
     * Therefore, this value is optional.
     * If it's not set (along with its corresponding bit in the field_mask -
     * @ref XCCL_MEM_MAP_PARAM_FIELD_ADDRESS), the @ref xccl_global_mem_map_start routine will consider
     * address as set to NULL and will allocate memory.
     */
    void     *address;
    size_t   length;
    int      root;
} xccl_mem_map_params_t;

xccl_status_t xccl_global_mem_map_start(xccl_team_h team, xccl_mem_map_params_t params,
                                        xccl_mem_h *memh_p);
xccl_status_t xccl_global_mem_map_test(xccl_mem_h memh_p);
xccl_status_t xccl_global_mem_unmap(xccl_mem_h memh_p);

END_C_DECLS

#endif
