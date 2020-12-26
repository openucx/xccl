#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

if HAVE_CUDA
SUBDIRS = kernels

sources = \
	cuda_mem_component.c \
	cuda_mem_component.h

component_noinst =
component_install = xccl_cuda_mem_component.la

xccl_cuda_mem_component_la_SOURCES =$(sources)
xccl_cuda_mem_component_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CPPFLAGS)
xccl_cuda_mem_component_la_LDFLAGS = -module -avoid-version $(CUDA_LDFLAGS)
xccl_cuda_mem_component_la_LIBADD = kernels/libxccl_cuda_kernels.la
pkglib_LTLIBRARIES = $(component_install)
endif
