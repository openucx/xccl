#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_NCCL
sources = \
	xccl_nccl_lib.c \
	xccl_nccl_lib.h

component_noinst =
component_install = xccl_team_lib_nccl.la

xccl_team_lib_nccl_la_SOURCES =$(sources)
xccl_team_lib_nccl_la_CPPFLAGS = $(AM_CPPFLAGS) $(NCCL_CPPFLAGS) $(CPPFLAGS)
xccl_team_lib_nccl_la_LDFLAGS = -module -avoid-version $(NCCL_LDFLAGS)
xccl_team_lib_nccl_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la

pkglib_LTLIBRARIES = $(component_install)
endif
