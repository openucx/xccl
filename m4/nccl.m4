#
# Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([XCCL_CHECK_NCCL],[

AS_IF([test "x$nccl_checked" != "xyes"],[

nccl_happy="no"

AC_ARG_WITH([nccl],
            [AS_HELP_STRING([--with-nccl=(DIR)], [Enable the use of NCCL (default is guess).])],
            [], [with_nccl=guess])

AS_IF([test "x$with_nccl" != "xno"],
    [save_CPPFLAGS="$CPPFLAGS"
     save_CFLAGS="$CFLAGS"
     save_LDFLAGS="$LDFLAGS"

     AS_IF([test ! -z "$with_nccl" -a "x$with_nccl" != "xyes" -a "x$with_nccl" != "xguess"],
            [
            xccl_check_nccl_dir="$with_nccl"
            xccl_check_nccl_libdir="$with_nccl/lib"
            CPPFLAGS="-I$with_nccl/include $save_CPPFLAGS $CUDA_CPPFLAGS"
            LDFLAGS="-L$xccl_check_nccl_libdir $save_LDFLAGS $CUDA_LDFLAGS"
            ])
        AS_IF([test ! -z "$with_nccl_libdir" -a "x$with_nccl_libdir" != "xyes"],
            [xccl_check_nccl_libdir="$with_nccl_libdir"
            LDFLAGS="-L$xccl_check_nccl_libdir $save_LDFLAGS $CUDA_LDFLAGS"])

        AC_CHECK_HEADERS([nccl.h],
            [AC_CHECK_LIB([nccl] , [ncclCommInitRank],
                           [nccl_happy="yes"],
                           [AC_MSG_WARN([NCCL is not detected. Disable.])
                            nccl_happy="no"])
            ], [nccl_happy="no"])


        AS_IF([test "x$nccl_happy" = "xyes"],
            [
                AC_SUBST(NCCL_CPPFLAGS, "-I$xccl_check_nccl_dir/include/ ")
                AC_SUBST(NCCL_LDFLAGS, "-lnccl -L$xccl_check_nccl_dir/lib")
            ],
            [
                AS_IF([test "x$with_nccl" != "xguess"],
                    [AC_MSG_ERROR([NCCL support is requested but NCCL packages cannot be found])],
                    [AC_MSG_WARN([NCCL not found])])
            ])
        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

    ],
    [AC_MSG_WARN([NCCL was explicitly disabled])])

nccl_checked=yes
AM_CONDITIONAL([HAVE_NCCL], [test "x$nccl_happy" != xno])
])

])
