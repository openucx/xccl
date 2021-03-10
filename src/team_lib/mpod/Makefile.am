if HAVE_MPOD
sources = \
	uthash.h \
	xccl_mpod_lib.h \
	xccl_mpod_lib.c \
	xccl_mpod_nccl_reqs.c \
	xccl_mpod_barrier.c \
	xccl_mpod_cpu.c \
	xccl_mpod_allreduce.c \
	xccl_mpod_allreduce_split.c \
	xccl_mpod_allreduce_coalesce.c \
	xccl_mpod_allreduce_replicate.c \
	xccl_mpod_bcast.c \
	xccl_mpod_allgather.c \
	xccl_mpod_alltoall.c \
	xccl_mpod_alltoallv.c

component_noinst =
component_install = xccl_team_lib_mpod.la

xccl_team_lib_mpod_la_SOURCES = $(sources)
xccl_team_lib_mpod_la_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS) -I../../../external
xccl_team_lib_mpod_la_LDFLAGS = -module -avoid-version
xccl_team_lib_mpod_la_LIBADD = $(XCCL_TOP_BUILDDIR)/src/libxccl.la

pkglib_LTLIBRARIES = $(component_install)
endif
