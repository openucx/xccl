FROM gitlab-master.nvidia.com:5005/dl/dgx/cuda:11.1-devel-sgodithi-ubuntu18.04-cuda11.1--1530168


ENV MOFED_VERSION 5.0-2.1.8.0
ENV MOFED_SITE_PLACE MLNX_OFED-${MOFED_VERSION}
ENV MOFED_IMAGE MLNX_OFED_LINUX-${MOFED_VERSION}-ubuntu18.04-x86_64
ENV HPCX_MOFED_VERSION 5.0-1.0.0.0
ENV HPCX_IMAGE hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-${HPCX_MOFED_VERSION}-ubuntu18.04-x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         wget \
         lsb-core \
         pciutils \
         libcap2 \
         libtool-bin \
         python \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

# Install Mellanox OFED
RUN wget http://content.mellanox.com/ofed/${MOFED_SITE_PLACE}/${MOFED_IMAGE}.tgz && \
    tar -xzvf ${MOFED_IMAGE}.tgz && \
    ${MOFED_IMAGE}/mlnxofedinstall --user-space-only --without-fw-update --all \
    -q --force --without-neohost-backend && \
    rm -rf ${MOFED_IMAGE} && \
    rm -rf *.tgz

# Install HPCX
RUN mkdir /tmp/hpcx && \
    cd /tmp/hpcx && \
    wget http://www.mellanox.com/downloads/hpc/hpc-x/v2.7/${HPCX_IMAGE}.tbz && \
    tar xjf ${HPCX_IMAGE}.tbz -C /opt/ && \
    rm -rf /tmp/hpcx

# Install UCX
RUN cd /opt && git clone https://github.com/openucx/ucx.git && cd ucx && \
    ./autogen.sh && \
    ./contrib/configure-release-mt --with-cuda=/usr/local/cuda/ --prefix=/opt/ucx/install && \
    make -j install

# Install XCCL
RUN cd /opt && git clone https://github.com/sergei-lebedev/mccl.git xccl && cd xccl && \
    git checkout topic/sra && \
    ./autogen.sh && \
    ./configure --with-cuda=/usr/local/cuda/ --with-ucx=/opt/ucx/install --prefix=/opt/xccl/install && \
    CPATH=/opt/ucx/install/include make -j install

# Install OMPI
RUN cd /opt && git clone https://github.com/open-mpi/ompi.git && cd ompi && \
    git checkout v4.0.x && \
    cp /opt/xccl/ompi_coll_xccl.patch . && \
    git apply ompi_coll_xccl.patch && \
    ./autogen.pl && \
    ./configure --prefix=$PWD/install --with-ucx=/opt/ucx/install --with-xccl=/opt/xccl/install --with-cuda=/usr/local/cuda && \
    make -j install

# Install OSU
RUN cd /opt && wget http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-5.6.3.tar.gz && \
    tar zxf osu-micro-benchmarks-5.6.3.tar.gz && \
    cd osu-micro-benchmarks-5.6.3 && \
    ./configure CC=/opt/ompi/install/bin/mpicc CXX=/opt/ompi/install/bin/mpicxx --enable-cuda --with-cuda-include=/usr/local/cuda/include --with-cuda-libpath=/usr/local/cuda/lib64 && \
    make -j
