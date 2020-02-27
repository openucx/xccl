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
	tccl_sharp_lib.c \
	tccl_sharp_lib.h \
	tccl_sharp_map.h \
	tccl_sharp_collective.c \
	tccl_sharp_collective.h

component_noinst =
component_install = tccl_team_lib_sharp.la

tccl_team_lib_sharp_la_SOURCES =$(sources)
tccl_team_lib_sharp_la_CPPFLAGS = $(AM_CPPFLAGS) $(SHARP_CPPFLAGS)
tccl_team_lib_sharp_la_LDFLAGS = -module -avoid-version $(SHARP_LDFLAGS)

pkglib_LTLIBRARIES = $(component_install)
endif
