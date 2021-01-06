#
# Copyright (c) 2020 Mellanox Technologies.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

CFLAGS = -I$(XCCL_PATH)/include -I$(UCX_PATH)/include
LDFLAGS = -L$(XCCL_PATH)/lib $(XCCL_PATH)/lib/libxccl.so $(UCX_PATH)/lib/libucs.so $(UCX_PATH)/lib/libucp.so -Wl,-rpath -Wl,$(XCCL_PATH)/lib -Wl,-rpath -Wl,$(XCCL_PATH)/lib

rel:
	mpicc -O3 -DNDEBUG -std=c11 $(CFLAGS) -o dpu_server dpu_server.c host_channel.c server_xccl.c $(LDFLAGS)

dbg:
	mpicc -O0 -g -std=c11 $(CFLAGS) -o dpu_server dpu_server.c host_channel.c server_xccl.c $(LDFLAGS)

clean:
	rm -f dpu_server
