#
# Copyright (c) 2009-2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

NVCC = nvcc
NVCCFLAGS = "${UCS_CPPFLAGS} -I${XCCL_TOP_SRCDIR}/src -I${XCCL_TOP_SRCDIR}/src/core" --compiler-options -fno-rtti,-fno-exceptions
NV_ARCH_FLAGS = -arch=sm_50 \
		-gencode=arch=compute_37,code=sm_37 \
		-gencode=arch=compute_50,code=sm_50 \
		-gencode=arch=compute_52,code=sm_52 \
		-gencode=arch=compute_60,code=sm_60 \
		-gencode=arch=compute_61,code=sm_61 \
		-gencode=arch=compute_70,code=sm_70 \
		-gencode=arch=compute_70,code=compute_70
LINK = $(LIBTOOL) --mode=link $(CC) -o $@

.cu.o:
	$(NVCC) -c $< -o $@  $(NVCCFLAGS) $(NV_ARCH_FLAGS)

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) -c  $< $(NVCCFLAGS) $(NV_ARCH_FLAGS)

comp_noinst = libxccl_cuda_kernels.la

libxccl_cuda_kernels_la_SOURCES  = xccl_cuda_reduce.cu \
                                   xccl_cuda_reduce_multi.cu \
                                   xccl_cuda_kernel.cu
libxccl_cuda_kernels_la_CPPFLAGS =

noinst_LTLIBRARIES = $(comp_noinst)
