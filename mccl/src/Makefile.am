#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
#

SUBDIRS =

lib_LTLIBRARIES  = libmccl.la
noinst_LIBRARIES =

libmccl_la_CPPFLAGS = $(AM_CPPFLAGS)
libmccl_la_CFLAGS   =
libmccl_la_LDFLAGS  = -version-info $(SOVERSION)

nobase_dist_libmccl_la_HEADERS = api/mccl.h

noinst_HEADERS = core/mccl_team.h \
				core/mccl_core.h \
				core/sbgp.h \
				core/mccl_inline.h \
				core/schedule.h

libmccl_la_SOURCES = core/mccl.c \
					core/sbgp.c \
					core/socketid.c \
					core/schedule.c \
					core/allreduce.c \
					core/barrier.c \
					core/bcast.c \
					core/schedule_bcast.c
libmccl_ladir = $(includedir)

libmccl_la_LIBADD = $(MCCL_TOP_BUILDDIR)/tccl/src/libtccl.la
