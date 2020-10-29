#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_TEAM_MRAIL
sources = \
	xccl_mrail_lib.h \
	xccl_mrail_lib.c

component_noinst =
component_install = xccl_team_lib_mrail.la

xccl_team_lib_mrail_la_SOURCES =$(sources)
xccl_team_lib_mrail_la_CPPFLAGS = $(AM_CPPFLAGS)
xccl_team_lib_mrail_la_LDFLAGS = -module -avoid-version
xccl_team_lib_mrail_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la
pkglib_LTLIBRARIES = $(component_install)
endif
