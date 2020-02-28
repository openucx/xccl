/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/


/**
 * Construct a XCCL version identifier from major and minor version numbers.
 */
#define XCCL_VERSION(_major, _minor) \
	(((_major) << XCCL_VERSION_MAJOR_SHIFT) | \
	 ((_minor) << XCCL_VERSION_MINOR_SHIFT))
#define XCCL_VERSION_MAJOR_SHIFT    24
#define XCCL_VERSION_MINOR_SHIFT    16


/**
 * XCCL API version is 1.0
 */
#define XCCL_API_MAJOR    1
#define XCCL_API_MINOR    0
#define XCCL_API_VERSION  XCCL_VERSION(1, 0)
