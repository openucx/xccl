/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef MCCL_TEAM_H
#define MCCL_TEAM_H
#include "api/tccl.h"
#include "mccl_core.h"

typedef struct sbgp_t sbgp_t;

typedef struct mccl_team_t {
    tccl_team_h tccl_team;
    sbgp_t *sbgp;
} mccl_team_t;
#endif
