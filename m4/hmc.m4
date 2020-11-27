#
# Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
# See file LICENSE for terms.
#

AC_DEFUN([XCCL_CHECK_HMC],[

AS_IF([test "x$hmc_checked" != "xyes"],[

hmc_happy="no"

AC_ARG_WITH([hmc],
            [AS_HELP_STRING([--with-hmc=(DIR)], [Enable the use of HMC (default is guess).])],
            [], [with_hmc=guess])

AS_IF([test "x$with_hmc" != "xno"],
    [save_CPPFLAGS="$CPPFLAGS"
     save_CFLAGS="$CFLAGS"
     save_LDFLAGS="$LDFLAGS"

     AS_IF([test ! -z "$with_hmc" -a "x$with_hmc" != "xyes" -a "x$with_hmc" != "xguess"],
            [
            xccl_check_hmc_dir="$with_hmc"
            AS_IF([test -d "$with_hmc/lib64"],[libsuff="64"],[libsuff=""])
            xccl_check_hmc_libdir="$with_hmc/lib$libsuff"
            CPPFLAGS="-I$with_hmc/include $save_CPPFLAGS"
            LDFLAGS="-L$xccl_check_hmc_libdir $save_LDFLAGS"
            ])
        AS_IF([test ! -z "$with_hmc_libdir" -a "x$with_hmc_libdir" != "xyes"],
            [xccl_check_hmc_libdir="$with_hmc_libdir"
            LDFLAGS="-L$xccl_check_hmc_libdir $save_LDFLAGS"])

        AC_CHECK_HEADERS([hmc.h],
            [AC_CHECK_LIB([hmc] , [hmc_init],
                           [hmc_happy="yes"],
                           [AC_MSG_WARN([HMC is not detected. Disable.])
                            hmc_happy="no"])
            ], [hmc_happy="no"])

        CFLAGS="$save_CFLAGS"
        CPPFLAGS="$save_CPPFLAGS"
        LDFLAGS="$save_LDFLAGS"

        AS_IF([test "x$hmc_happy" = "xyes"],
            [
                AC_SUBST(HMC_CPPFLAGS, "-I$xccl_check_hmc_dir/include/ ")
                AC_SUBST(HMC_LDFLAGS, "-L$xccl_check_hmc_dir/lib -lhmc")
            ],
            [
                AS_IF([test "x$with_hmc" != "xguess"],
                    [AC_MSG_ERROR([HMC support is requested but HMC packages cannot be found])],
                    [AC_MSG_WARN([HMC not found])])
            ])

    ],
    [AC_MSG_WARN([HMC was explicitly disabled])])

hmc_checked=yes
AM_CONDITIONAL([HAVE_HMC], [test "x$hmc_happy" != xno])
])

])
