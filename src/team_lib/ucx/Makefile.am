#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_UCX
allgather = allgather/allgather.h \
	allgather/allgather_ring.c

allreduce = allreduce/allreduce.h \
    allreduce/allreduce_knomial.h \
    allreduce/allreduce.c \
	allreduce/allreduce_knomial.c \
    allreduce/allreduce_sra.c

alltoall = alltoall/alltoall.h \
	alltoall/alltoall_pairwise.c \
	alltoall/alltoall_linear_shift.c

alltoallv = alltoallv/alltoallv.h \
	alltoallv/alltoallv_pairwise.c

reduce = reduce.h \
	reduce/reduce_linear.c \
	reduce/reduce_knomial.c

fanin = fanin.h \
	fanin/fanin_linear.c

fanout = fanout.h \
	fanout/fanout_linear.c

bcast = bcast.h \
	bcast/bcast_linear.c\
	bcast/bcast_knomial.c

barrier = barrier.h \
	barrier/barrier_knomial.c

sources = \
	xccl_ucx_tag.h \
	xccl_ucx_team.h \
	xccl_ucx_team.c \
	xccl_ucx_context.h \
	xccl_ucx_context.c \
	xccl_ucx_lib.h \
	xccl_ucx_ep.h \
	xccl_ucx_lib.c \
	xccl_ucx_sendrecv.h \
	${allgather} \
	${allreduce} \
	${alltoall} \
	${alltoallv} \
	${reduce} \
	${fanin}\
	${fanout}\
	${bcast} \
	${barrier}

component_noinst =
component_install = xccl_team_lib_ucx.la

xccl_team_lib_ucx_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la
xccl_team_lib_ucx_la_SOURCES =$(sources)
xccl_team_lib_ucx_la_CPPFLAGS = $(AM_CPPFLAGS) $(UCX_CPPFLAGS)
xccl_team_lib_ucx_la_LDFLAGS = -module -avoid-version $(UCX_LDFLAGS) $(UCX_LIBADD)

pkglib_LTLIBRARIES = $(component_install)
endif
