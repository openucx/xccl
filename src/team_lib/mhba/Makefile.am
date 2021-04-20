#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

sources =                   \
	xccl_mhba_lib.c         \
	xccl_mhba_collective.c  \
	xccl_mhba_lib.h         \
	xccl_mhba_mkeys.c       \
	xccl_mhba_mkeys.h       \
	xccl_mhba_ib.c          \
	xccl_mhba_team.c        \
	xccl_mhba_ib.h          \
	stdatomic.h \
	xccl_mhba_socket_comm.c \
	xccl_mhba_socket_comm.h

component_noinst =
component_install = xccl_team_lib_mhba.la

xccl_team_lib_mhba_la_SOURCES =$(sources)
xccl_team_lib_mhba_la_CPPFLAGS = $(AM_CPPFLAGS) $(MHBA_CPPFLAGS) $(CPPFLAGS)
xccl_team_lib_mhba_la_LDFLAGS = -module -avoid-version $(MHBA_LDFLAGS)
xccl_team_lib_mhba_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la -libverbs -lmlx5

pkglib_LTLIBRARIES = $(component_install)
