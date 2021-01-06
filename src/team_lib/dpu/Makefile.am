#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_DPU
sources = \
	xccl_dpu_lib.c \
	xccl_dpu_lib.h

component_noinst =
component_install = xccl_team_lib_dpu.la

xccl_team_lib_dpu_la_SOURCES =$(sources)
xccl_team_lib_dpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) $(UCX_CPPFLAGS)
xccl_team_lib_dpu_la_LDFLAGS = -module -avoid-version $(UCX_LDFLAGS) $(UCX_LIBADD)
xccl_team_lib_dpu_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la

pkglib_LTLIBRARIES = $(component_install)
endif
