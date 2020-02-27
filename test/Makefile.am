#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS = test_mpi_allreduce test_mpi_bcast test_mpi_barrier

test_mpi_allreduce_SOURCES=test_mpi_allreduce.c test_mpi.c
test_mpi_bcast_SOURCES=test_mpi_bcast.c test_mpi.c
test_mpi_barrier_SOURCES=test_mpi_barrier.c test_mpi.c

CC=mpicc
CFLAGS+=-I${includedir} -std=c11

LDFLAGS=-L$(libdir) -ltccl
