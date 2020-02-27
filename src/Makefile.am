#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

SUBDIRS = 
if HAVE_UCX
SUBDIRS += team_lib/ucx
endif
if HAVE_SHARP
SUBDIRS += team_lib/sharp
endif
if HAVE_VMC
SUBDIRS += team_lib/vmc
endif

lib_LTLIBRARIES  = libtccl.la
noinst_LIBRARIES =

libtccl_la_CPPFLAGS = $(AM_CPPFLAGS)
libtccl_la_CFLAGS   =
libtccl_la_LDFLAGS  = -ldl -version-info $(SOVERSION) --no-undefined

nobase_dist_libtccl_la_HEADERS = \
	api/tccl.h                   \
	api/tccl_def.h               \
	api/tccl_version.h           \
	api/tccl_status.h

noinst_HEADERS = core/tccl_team_lib.h

libtccl_la_SOURCES =      \
	core/tccl_init.c      \
	core/tccl_finalize.c  \
	core/tccl_context.c   \
	core/tccl_team.c      \
	core/tccl_collective.c\
	core/tccl_query.c     \
	utils/socketid.c

libtccl_ladir = $(includedir)

