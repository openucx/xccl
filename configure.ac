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

define([mccl_ver_major], 1)
define([mccl_ver_minor], 0)
define([mccl_ver_micro], esyscmd([sh -c "git rev-list HEAD | wc -l | sed -e 's/ *//g' | xargs -n1 printf"]))
define([mccl_git_sha], esyscmd([sh -c "git rev-parse --short HEAD"]))

# This is the API version (see libtool library versioning)
# http://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
# current:rev:age
define([libmccl_so_version], 1:0:0)

AC_INIT([mccl], [mccl_ver_major.mccl_ver_minor])
AC_USE_SYSTEM_EXTENSIONS
AC_GNU_SOURCE
AC_CONFIG_HEADERS([config.h])

MCCL_TOP_BUILDDIR="`pwd`"
AC_SUBST(MCCL_TOP_BUILDDIR)

cd "$srcdir"
MCCL_TOP_SRCDIR="`pwd`"
AC_SUBST(MCCL_TOP_SRCDIR)
cd "$MCCL_TOP_BUILDDIR"

AC_MSG_NOTICE([builddir: $MCCL_TOP_BUILDDIR])
AC_MSG_NOTICE([srcdir: $MCCL_TOP_SRCDIR])

AM_INIT_AUTOMAKE([1.10 foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])
AM_MAINTAINER_MODE
AC_CONFIG_MACRO_DIR([m4])

define([mccl_prefix], [/opt/mellanox/mccl])
AC_PREFIX_DEFAULT([mccl_prefix])

RPM_RELEASE=1
MAJOR_VERSION=mccl_ver_major
MINOR_VERSION=mccl_ver_minor
MICRO_VERSION=mccl_ver_micro
UCH_GIT_SHA=mccl_git_sha
VERSION=$MAJOR_VERSION.$MINOR_VERSION.$MICRO_VERSION
SOVERSION=libmccl_so_version
AC_SUBST(MAJOR_VERSION)
AC_SUBST(MINOR_VERSION)
AC_SUBST(MICRO_VERSION)
AC_SUBST(RPM_RELEASE)
AC_SUBST(SOVERSION)
AC_SUBST(UCH_GIT_SHA)
AC_MSG_RESULT([git sha: $mccl_git_sha])

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
else
	CFLAGS="$CFLAGS -O3 -DNDEBUG"
fi


includes="-I${MCCL_TOP_SRCDIR}/tccl/src -I${MCCL_TOP_SRCDIR}/mccl/src/api -I${MCCL_TOP_SRCDIR}/mccl/src"
CPPFLAGS="$CPPFLAGS $includes -std=gnu11"

AC_CONFIG_FILES([
                 Makefile
                 mccl/Makefile
                 mccl/src/Makefile
                 mccl/test/Makefile
                 ])
AC_CONFIG_SUBDIRS([tccl])
AC_OUTPUT
