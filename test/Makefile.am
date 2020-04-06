#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

bin_PROGRAMS =  test_mpi_allreduce \
				test_mpi_bcast \
				test_mpi_barrier \
				test_mpi_fanout_get \
				test_mpi_create_team_nb
if HAVE_CUDA
bin_PROGRAMS += test_mpi_cuda_allreduce \
                test_mpi_cuda_bcast \
				test_mpi_cuda_reduce
endif

test_mpi_allreduce_SOURCES=test_mpi_allreduce.c test_mpi.c
test_mpi_bcast_SOURCES=test_mpi_bcast.c test_mpi.c
test_mpi_barrier_SOURCES=test_mpi_barrier.c test_mpi.c
test_mpi_fanout_get_SOURCES=test_mpi_fanout_get.c test_mpi.c
test_mpi_create_team_nb_SOURCES=test_mpi_create_team_nb.c test_mpi.c

if HAVE_CUDA
test_mpi_cuda_allreduce_SOURCES=test_mpi_cuda_allreduce.c test_mpi.c
test_mpi_cuda_bcast_SOURCES=test_mpi_cuda_bcast.c test_mpi.c
test_mpi_cuda_reduce_SOURCES=test_mpi_cuda_reduce.c test_mpi.c
endif

CC=mpicc
CFLAGS+=-I${includedir} -std=c11 

LDFLAGS=-L$(libdir) -lxccl

if HAVE_CUDA
CFLAGS += $(CUDA_CFLAGS)
LDFLAGS+= $(CUDA_LDFLAGS) -lcudart
endif
