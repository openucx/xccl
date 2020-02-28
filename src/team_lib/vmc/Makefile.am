#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_VMC
sources = \
	xccl_vmc_lib.c \
	xccl_vmc_lib.h

component_noinst =
component_install = xccl_team_lib_vmc.la

xccl_team_lib_vmc_la_SOURCES =$(sources)
xccl_team_lib_vmc_la_CPPFLAGS = $(AM_CPPFLAGS) $(VMC_CPPFLAGS)
xccl_team_lib_vmc_la_LDFLAGS = -module -avoid-version  $(VMC_LDFLAGS)
pkglib_LTLIBRARIES = $(component_install)
endif
