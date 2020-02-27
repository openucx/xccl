/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/


/**
 * Construct a TCCL version identifier from major and minor version numbers.
 */
#define TCCL_VERSION(_major, _minor) \
	(((_major) << TCCL_VERSION_MAJOR_SHIFT) | \
	 ((_minor) << TCCL_VERSION_MINOR_SHIFT))
#define TCCL_VERSION_MAJOR_SHIFT    24
#define TCCL_VERSION_MINOR_SHIFT    16


/**
 * TCCL API version is 1.0
 */
#define TCCL_API_MAJOR    1
#define TCCL_API_MINOR    0
#define TCCL_API_VERSION  TCCL_VERSION(1, 0)
