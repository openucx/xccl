/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#define _GNU_SOURCE
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>
#include <api/tccl_status.h>

#define UNDEFSOCKET -2
#define NOSOCKET    -1
typedef unsigned long int cpu_mask_t;
#define NCPUBITS ( 8 * sizeof(cpu_mask_t))

#define CPUELT(cpu)  ((cpu) / NCPUBITS)
#define CPUMASK(cpu) ((cpu_mask_t) 1 << ((cpu) % NCPUBITS))

#define SBGP_CPU_ISSET(cpu, setsize, cpusetp)    \
    ({  size_t __cpu = (cpu);               \
        __cpu < 8 * (setsize)               \
        ? ((((const cpu_mask_t *) ((cpusetp)->__bits))[__CPUELT(__cpu)] \
                & CPUMASK (__cpu))) != 0    \
        : 0; })

static int parse_cpuset_file(FILE *file, int* nr_psbl_cpus) {
    unsigned long start, stop;
    while(fscanf(file, "%lu", &start) == 1) {
        int c = fgetc(file);
        stop = start;
        if (c == '-') {
            if (fscanf(file, "%lu", &stop) != 1) {
                /* Range is usually <int>-<int> */
                errno = EINVAL;
                return -1;
            }
            c = fgetc(file);
        }

        if (c == EOF || c == '\n') {
            *nr_psbl_cpus = (int) stop + 1;
            break;
        }

        if (c != ',') {
            /* Wrong terminating char */
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

tccl_status_t tccl_get_bound_socket_id(int *socketid) {
    int err, nr_cpus=0, nr_onl_cpus=0, nr_psbl_cpus=0, try=1000, i=0, pos = 0;
    int max_sockets = 64;
    unsigned cpu;
    size_t setsize;
    cpu_set_t *cpuset = NULL;
    FILE *fptr, *possible;
    char str[1024], ch[64];
    long tmp_socketid = NOSOCKET, tmpid = NOSOCKET, *socket_ids;
    unsigned long socket_bitmap = 0,
                  socket_bitmap_all = 0,
                  bitcheck = 1;

    socket_ids = malloc(max_sockets * sizeof(long));
    if (socket_ids == NULL) {
        return TCCL_ERR_NO_MESSAGE;
    }

    memset(socket_ids, -1, max_sockets * sizeof(long));

    /* Get the number of total procs and online procs */
    nr_cpus = sysconf(_SC_NPROCESSORS_CONF);

    /* Need to make sure nr_cpus !< possible_cpus+1 */
    possible = fopen("/sys/devices/system/cpu/possible", "r");
    if (possible) {
        if (parse_cpuset_file(possible, &nr_psbl_cpus) == 0) {
            if (nr_cpus < nr_psbl_cpus+1)
                nr_cpus = nr_psbl_cpus;
        }
        fclose(possible);
    }

    if (!nr_cpus) {
        free(socket_ids);
        return TCCL_ERR_NO_MESSAGE;
    }

    /* The cpuset size on some kernels needs to be bigger than
     * the number of nr_cpus, hwloc gets around this
     * by blocking on a loop and increasing nr_cpus.
     * We will try 1000 (arbitrary) attempts, and revert to hwloc
     * if all fail */
    setsize = ((nr_cpus + NCPUBITS - 1) / NCPUBITS) * sizeof(cpu_mask_t);
    cpuset = __sched_cpualloc(nr_cpus);
    if (NULL == cpuset) {
        free(socket_ids);
        return TCCL_ERR_NO_MESSAGE;
    }

    while (0 < sched_getaffinity(0, setsize, cpuset) && try>0) {
        __sched_cpufree(cpuset);
        try--;
        nr_cpus*=2;
        cpuset = __sched_cpualloc(nr_cpus);
        if (NULL == cpuset) {
            try = 0;
            break;
        }
        setsize = ((nr_cpus + NCPUBITS - 1) / NCPUBITS) * sizeof(cpu_mask_t);
    }

    /* If after all tries we're still not getting it, error out
     * let hwloc take over */
    if (try == 0) {
        fprintf(stderr, "Error when manually trying to discover socket_id using sched_getaffinity()\n");
        __sched_cpufree(cpuset);
        free(socket_ids);
        return TCCL_ERR_NO_MESSAGE;
    }

    /* Loop through all cpus, and check if I'm bound to the socket */
    for (cpu = 0; cpu < nr_cpus; cpu++) {
        sprintf(str,"/sys/bus/cpu/devices/cpu%d/topology/physical_package_id", cpu);
        fptr = fopen(str,"r");
        if (!fptr) {
            /* Do nothing just skip */
            continue;
        }

        /* Read socketid and convert to int */
        i=0;
        ch[i] = (char) getc(fptr);
        while ((ch[i] != '\n') && (ch[i] != EOF)) {
            i++;
            ch[i] = (char) getc(fptr);
        }

        /* Mark the id */
        tmpid = strtol(ch, NULL, 10);
        for (i = 0; i < max_sockets; i++) {
            if (socket_ids[i] == -1) {
                /* new entry */
                socket_ids[i] = tmpid;
                tmpid = i;
                break;
            }
            else if (socket_ids[i] == tmpid) {
                /* existing entry, update position */
                tmpid = i;
                break;
            }
        }

        /* Check if we need more memory */
        if (i == max_sockets && tmpid != max_sockets - 1) {
            socket_ids = realloc(socket_ids, max_sockets*=2);
            if (socket_ids == NULL) {
                return TCCL_ERR_NO_MESSAGE;
            }
            memset(socket_ids+i, -1, (max_sockets - i) * sizeof(long));
            socket_ids[i] = tmpid;
            tmpid = i;
        }

        socket_bitmap_all = socket_bitmap_all | (1 << tmpid);

        /* Set socket bit */
        if (SBGP_CPU_ISSET(cpu, setsize, cpuset)) {
            tmp_socketid = tmpid;
            socket_bitmap = socket_bitmap | (1 << tmp_socketid);
        }
        fclose(fptr);
    }

    /* Either set the socket_id or check if we're bound to more than 2 */
    if (__builtin_popcountl(socket_bitmap) > 1) {
        *socketid = NOSOCKET;
    }
    else {
        *socketid = (int) tmp_socketid;
    }

    __sched_cpufree(cpuset);
    free(socket_ids);

    if (*socketid == NOSOCKET) {
        return TCCL_OK;
    }

    /* Some archs (eg. POWER) seem to have non-linear socket_ids.
     * Convert to logical index */
    for (i = 0; i < *socketid; i++) {
        if (socket_bitmap_all & bitcheck) {
            pos++;
        }
        bitcheck = bitcheck << 1;
    }

    *socketid = pos;
    return TCCL_OK;
}
