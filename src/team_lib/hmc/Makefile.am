#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_HMC
sources = \
	xccl_hmc_lib.c \
	xccl_hmc_lib.h

component_noinst =
component_install = xccl_team_lib_hmc.la

xccl_team_lib_hmc_la_SOURCES =$(sources)
xccl_team_lib_hmc_la_CPPFLAGS = $(AM_CPPFLAGS) $(HMC_CPPFLAGS)
xccl_team_lib_hmc_la_LDFLAGS = -module -avoid-version  $(HMC_LDFLAGS)
xccl_team_lib_hmc_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la
pkglib_LTLIBRARIES = $(component_install)
endif
