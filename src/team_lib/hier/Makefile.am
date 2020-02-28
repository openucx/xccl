#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources = \
	tccl_hier_lib.h \
	tccl_hier_lib.c \
	tccl_hier_context.h \
	tccl_hier_context.c

component_noinst =
component_install = tccl_team_lib_hier.la

tccl_team_lib_hier_la_SOURCES =$(sources)
tccl_team_lib_hier_la_CPPFLAGS = $(AM_CPPFLAGS)
tccl_team_lib_hier_la_LDFLAGS = -module -avoid-version
pkglib_LTLIBRARIES = $(component_install)

