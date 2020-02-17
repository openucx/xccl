/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_TEST_H_
#define MCCL_TEST_H_
#include "api/mccl.h"

extern mccl_comm_h mccl_comm_world;
int mccl_test_init(int argc, char **argv, uint64_t caps);
int mccl_test_fini(void);
#endif
