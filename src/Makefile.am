#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

SUBDIRS =              \
    .                  \
	team_lib/hier      \
	team_lib/multirail

if HAVE_UCX
SUBDIRS += team_lib/ucx
endif
if HAVE_SHARP
SUBDIRS += team_lib/sharp
endif
if HAVE_VMC
SUBDIRS += team_lib/vmc
endif
if HAVE_CUDA
SUBDIRS += utils/cuda
endif

lib_LTLIBRARIES  = libxccl.la
noinst_LIBRARIES =

libxccl_la_CPPFLAGS = $(AM_CPPFLAGS)
libxccl_la_CFLAGS   =
libxccl_la_LDFLAGS  = -ldl -version-info $(SOVERSION) --no-undefined

nobase_dist_libxccl_la_HEADERS = \
	api/xccl.h                   \
	api/xccl_def.h               \
	api/xccl_version.h           \
	api/xccl_status.h            \
	api/xccl_tls.h

noinst_HEADERS = core/xccl_team_lib.h

libxccl_la_SOURCES =       \
	core/xccl_init.c       \
	core/xccl_finalize.c   \
	core/xccl_lib.c        \
	core/xccl_context.c    \
	core/xccl_team.c       \
	core/xccl_collective.c \
	core/xccl_mm.c         \
	core/xccl_query.c      \
	core/xccl_global_opts.c\
	core/xccl_team_lib.c   \
	core/xccl_schedule.c   \
	utils/socketid.c       \
	utils/mem_component.c

libxccl_ladir = $(includedir)

install-exec-hook:
	cp -f $(XCCL_TOP_BUILDDIR)/src/libxccl.la $(libdir)/
	perl -pi -e "s/installed=no/installed=yes/" $(libdir)/libxccl.la
