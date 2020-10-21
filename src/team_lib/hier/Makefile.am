#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
	xccl_hier_lib.h \
	xccl_hier_lib.c \
	xccl_hier_context.h \
	xccl_hier_context.c \
	xccl_hier_team.h \
	xccl_hier_team.c \
	xccl_hier_schedule.h \
	xccl_hier_schedule.c \
	xccl_hier_schedule_allreduce.c \
	xccl_hier_schedule_barrier.c \
	xccl_hier_schedule_bcast.c

component_noinst =
component_install = xccl_team_lib_hier.la

xccl_team_lib_hier_la_SOURCES =$(sources)
xccl_team_lib_hier_la_CPPFLAGS = $(AM_CPPFLAGS)
xccl_team_lib_hier_la_LDFLAGS = -module -avoid-version
xccl_team_lib_hier_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la
pkglib_LTLIBRARIES = $(component_install)

