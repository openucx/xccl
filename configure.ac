#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
# This software product is a proprietary product of Mellanox Technologies Ltd.
# (the "Company") and all right, title, and interest and to the software product,
# including all associated intellectual property rights, are and shall
# remain exclusively with the Company.
#
# This software product is governed by the End User License Agreement
# provided with the software product.
# $COPYRIGHT$
# $HEADER$

define([xccl_ver_major], 1)
define([xccl_ver_minor], 0)
define([xccl_ver_micro], esyscmd([sh -c "git rev-list HEAD | wc -l | sed -e 's/ *//g' | xargs -n1 printf"]))
define([xccl_git_sha], esyscmd([sh -c "git rev-parse --short HEAD"]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libxccl_so_version], 1:0:0)
AC_INIT([xccl], [xccl_ver_major.xccl_ver_minor])
: ${CPPFLAGS=""}
: ${CFLAGS=""}
: ${CXXFLAGS=""}

AC_CONFIG_HEADERS([config.h])

XCCL_TOP_BUILDDIR="`pwd`"
AC_SUBST(XCCL_TOP_BUILDDIR)

cd "$srcdir"
XCCL_TOP_SRCDIR="`pwd`"
AC_SUBST(XCCL_TOP_SRCDIR)
cd "$XCCL_TOP_BUILDDIR"

AC_MSG_NOTICE([builddir: $XCCL_TOP_BUILDDIR])
AC_MSG_NOTICE([srcdir: $XCCL_TOP_SRCDIR])
AM_INIT_AUTOMAKE([1.10 foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([m4])

define([xccl_prefix], [/opt/mellanox/xccl])
AC_PREFIX_DEFAULT([xccl_prefix])

RPM_RELEASE=1
MAJOR_VERSION=xccl_ver_major
MINOR_VERSION=xccl_ver_minor
MICRO_VERSION=xccl_ver_micro
XCCL_GIT_SHA=xccl_git_sha
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION
SOVERSION=libxccl_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(XCCL_GIT_SHA)
AC_MSG_RESULT([git sha: $xccl_git_sha])

CFLAGS_save="$CFLAGS"
AC_PROG_CC
AC_PROG_CXX
AM_PROG_AS
AC_PROG_INSTALL
AC_PROG_LIBTOOL
AC_HEADER_STDC
CFLAGS="$CFLAGS_save"

AC_ARG_ENABLE([debug],
AS_HELP_STRING([--enable-debug], [Enable extra debugging code (default is NO).]),
[], [enable_debug=no])

if test $enable_debug = yes; then
    AC_DEFINE([ENABLE_DEBUG], [1], [Enable debugging code])
    CFLAGS="$CFLAGS -O0 -g3"
    AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_TRACE_POLL], [Highest log level])
else
    AC_DEFINE([UCS_MAX_LOG_LEVEL], [UCS_LOG_LEVEL_INFO], [Highest log level])
    CFLAGS="$CFLAGS -O3 -DNDEBUG"
fi

AC_ARG_ENABLE([team-mrail],
AS_HELP_STRING([--enable-team-mrail], [Enable multirail team (default is NO).]),
[], [enable_team_mrail=no])
AM_CONDITIONAL([HAVE_TEAM_MRAIL], [test "x$enable_team_mrail" = xyes])

AM_CONDITIONAL([HAVE_UCX], [false])
m4_include([m4/ucx.m4])
XCCL_CHECK_UCX

if test $ucx_happy != "yes"; then
   AC_MSG_ERROR([UCX is not available])
fi
AC_MSG_RESULT([UCX support: $ucx_happy])

AM_CONDITIONAL([HAVE_SHARP], [false])
m4_include([m4/sharp.m4])
XCCL_CHECK_SHARP

AC_MSG_RESULT([SHARP support: $sharp_happy])

AM_CONDITIONAL([HAVE_HMC], [false])
m4_include([m4/hmc.m4])
XCCL_CHECK_HMC

AC_MSG_RESULT([HMC support: $hmc_happy])

AC_ARG_WITH([cuda],
            AC_HELP_STRING([--with-cuda=PATH], [Path to non-standard CUDA installation]),
            [AS_IF([test -d $withval/lib64], [cuda_libdir="lib64"], [cuda_libdir="lib"])
             AC_SUBST(CUDA_CPPFLAGS, "-I$withval/include")
             AC_SUBST(CUDA_LDFLAGS, "-L$withval/$cuda_libdir -lcudart")
             cuda_happy="yes"],
            [cuda_happy="no"])
AM_CONDITIONAL([HAVE_CUDA], [test "x$cuda_happy" != xno])
AC_MSG_RESULT([CUDA support: $cuda_happy; $CUDA_CPPFLAGS $CUDA_LDFLAGS])

AC_ARG_WITH([dpu],
            AC_HELP_STRING([--with-dpu=yes/no], [Enable/Disable dpu team]),
            [AS_IF([test "x$with_dpu" != "xno"], dpu_happy="yes", dpu_happy="no")],
            [dpu_happy="no"])
AM_CONDITIONAL([HAVE_DPU], [test "x$dpu_happy" != xno])
AC_MSG_RESULT([DPU support: $dpu_happy])

AM_CONDITIONAL([HAVE_NCCL], [false])
if test "x$cuda_happy" != xno; then
    m4_include([m4/nccl.m4])
    XCCL_CHECK_NCCL
    AC_MSG_RESULT([NCCL support: $nccl_happy])
fi

includes="-I${XCCL_TOP_SRCDIR}/src -I${XCCL_TOP_SRCDIR}/src/core -I${XCCL_TOP_SRCDIR}/src/utils"
CPPFLAGS="$UCS_CPPFLAGS $CPPFLAGS $includes"
LDFLAGS="$LDFLAGS $UCS_LDFLAGS $UCS_LIBADD"
AC_CONFIG_FILES([
                 Makefile
                 src/Makefile
                 src/team_lib/ucx/Makefile
                 src/team_lib/sharp/Makefile
                 src/team_lib/hmc/Makefile
                 src/team_lib/hier/Makefile
                 src/team_lib/multirail/Makefile
                 src/team_lib/nccl/Makefile
                 src/team_lib/dpu/Makefile
                 src/utils/cuda/Makefile
                 src/utils/cuda/kernels/Makefile
                 test/Makefile
                 ])
AC_OUTPUT
