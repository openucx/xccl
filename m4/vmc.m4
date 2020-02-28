#
# Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([XCCL_CHECK_VMC],[

AS_IF([test "x$vmc_checked" != "xyes"],[

vmc_happy="no"

AC_ARG_WITH([vmc],
            [AS_HELP_STRING([--with-vmc=(DIR)], [Enable the use of VMC (default is guess).])],
            [], [with_vmc=guess])

AS_IF([test "x$with_vmc" != "xno"],
    [save_CPPFLAGS="$CPPFLAGS"
     save_CFLAGS="$CFLAGS"
     save_LDFLAGS="$LDFLAGS"

     AS_IF([test ! -z "$with_vmc" -a "x$with_vmc" != "xyes" -a "x$with_vmc" != "xguess"],
            [
            xccl_check_vmc_dir="$with_vmc"
            AS_IF([test -d "$with_vmc/lib64"],[libsuff="64"],[libsuff=""])
            xccl_check_vmc_libdir="$with_vmc/lib$libsuff"
            CPPFLAGS="-I$with_vmc/include $save_CPPFLAGS"
            LDFLAGS="-L$xccl_check_vmc_libdir $save_LDFLAGS"
            ])
        AS_IF([test ! -z "$with_vmc_libdir" -a "x$with_vmc_libdir" != "xyes"],
            [xccl_check_vmc_libdir="$with_vmc_libdir"
            LDFLAGS="-L$xccl_check_vmc_libdir $save_LDFLAGS"])

        AC_CHECK_HEADERS([vmc.h],
            [AC_CHECK_LIB([vmc] , [vmc_init],
                           [vmc_happy="yes"],
                           [AC_MSG_WARN([VMC is not detected. Disable.])
                            vmc_happy="no"])
            ], [vmc_happy="no"])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

        AS_IF([test "x$vmc_happy" = "xyes"],
            [
                AC_SUBST(VMC_CPPFLAGS, "-I$xccl_check_vmc_dir/include/ ")
                AC_SUBST(VMC_LDFLAGS, "-L$xccl_check_vmc_dir/lib -lvmc")
            ],
            [
                AS_IF([test "x$with_vmc" != "xguess"],
                    [AC_MSG_ERROR([VMC support is requested but VMC packages cannot be found])],
                    [AC_MSG_WARN([VMC not found])])
            ])

    ],
    [AC_MSG_WARN([VMC was explicitly disabled])])

vmc_checked=yes
AM_CONDITIONAL([HAVE_VMC], [test "x$vmc_happy" != xno])
])

])
