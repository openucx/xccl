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
				test_mpi_reduce \
				test_mpi_fanout_get \
				test_mpi_create_team_nb \
				test_mpi_alltoall \
				test_mpi_alltoallv \
				test_mpi_allgather \
				test_mpi_mt
if HAVE_CUDA
bin_PROGRAMS += test_mpi_cuda_allreduce \
                test_mpi_cuda_bcast \
				test_mpi_cuda_reduce
endif

test_mpi_allreduce_SOURCES=test_mpi_allreduce.c test_mpi.c
test_mpi_reduce_SOURCES=test_mpi_reduce.c test_mpi.c
test_mpi_bcast_SOURCES=test_mpi_bcast.c test_mpi.c
test_mpi_barrier_SOURCES=test_mpi_barrier.c test_mpi.c
test_mpi_fanout_get_SOURCES=test_mpi_fanout_get.c test_mpi.c
test_mpi_create_team_nb_SOURCES=test_mpi_create_team_nb.c test_mpi.c
test_mpi_alltoall_SOURCES=test_mpi_alltoall.c test_mpi.c
test_mpi_alltoallv_SOURCES=test_mpi_alltoallv.c test_mpi.c
test_mpi_allgather_SOURCES=test_mpi_allgather.c test_mpi.c
test_mpi_mt_SOURCES=test_mt.c test_mpi.c
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
