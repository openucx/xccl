/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_TEST_H_
#define MCCL_TEST_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include "api/mccl.h"

#define STR(x) # x
#define MCCL_CHECK(_call) if (MCCL_SUCCESS != (_call)) {\
        fprintf(stderr, "fail: %s\n", STR(_call)); \
        exit(-1);                                  \
    }

extern mccl_comm_h mccl_comm_world;
int mccl_test_init(int argc, char **argv, uint64_t caps);
int mccl_test_fini(void);
#endif
