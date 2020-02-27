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
	tccl_vmc_lib.c \
	tccl_vmc_lib.h

component_noinst =
component_install = tccl_team_lib_vmc.la

tccl_team_lib_vmc_la_SOURCES =$(sources)
tccl_team_lib_vmc_la_CPPFLAGS = $(AM_CPPFLAGS) $(VMC_CPPFLAGS)
tccl_team_lib_vmc_la_LDFLAGS = -module -avoid-version  $(VMC_LDFLAGS)
pkglib_LTLIBRARIES = $(component_install)
endif
