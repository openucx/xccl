#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_SHARP
sources = \
	xccl_sharp_lib.c \
	xccl_sharp_lib.h \
	xccl_sharp_map.h \
	xccl_sharp_collective.c \
	xccl_sharp_collective.h

component_noinst =
component_install = xccl_team_lib_sharp.la

xccl_team_lib_sharp_la_SOURCES =$(sources)
xccl_team_lib_sharp_la_CPPFLAGS = $(AM_CPPFLAGS) $(SHARP_CPPFLAGS)
xccl_team_lib_sharp_la_LDFLAGS = -module -avoid-version $(SHARP_LDFLAGS)
xccl_team_lib_sharp_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la

pkglib_LTLIBRARIES = $(component_install)
endif
