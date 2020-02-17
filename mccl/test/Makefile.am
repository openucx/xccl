#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = ar_corr ar_bench bcast_corr

ar_corr_SOURCES=allreduce_correctness.c mccl_test.c
ar_bench_SOURCES=allreduce_benchmark.c mccl_test.c
bcast_corr_SOURCES=bcast_correctness.c mccl_test.c

CC=mpicc
CFLAGS+=-I${includedir} -std=c11

LDFLAGS=-L$(libdir) -lmccl
