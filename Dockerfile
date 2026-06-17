FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake git gfortran ccache \
    libopenmpi-dev libtbb-dev libboost-all-dev \
    libhwloc-dev curl pkg-config automake autoconf libtool make \
    && rm -rf /var/lib/apt/lists/*

ENV CCACHE_DIR=/ccache
RUN ccache -M 15G
ENV PATH="/usr/lib/ccache:$PATH"

ENV CXXFLAGS="-march=cascadelake -O3"
ENV CFLAGS="-march=cascadelake -O3"

RUN --mount=type=cache,target=/ccache \
    git clone -b v3.24.0 --single-branch https://gitlab.com/petsc/petsc.git /opt/petsc-src && \
    cd /opt/petsc-src && \
    ./configure --with-cc=mpicc --with-cxx=mpicxx --with-fc=mpif90 \
                --with-debugging=0 --with-scalar-type=real \
                --download-fblaslapack=1 --prefix=/opt/petsc \
                COPTFLAGS="$CFLAGS" CXXOPTFLAGS="$CXXFLAGS" && \
    make -j$(nproc) all && make install

RUN --mount=type=cache,target=/ccache \
    git clone https://github.com/cburstedde/p4est.git /opt/p4est-src && \
    cd /opt/p4est-src && \
    git submodule init && git submodule update && \
    ./bootstrap && mkdir /opt/p4est-build && cd /opt/p4est-build && \
    ../p4est-src/configure --enable-mpi --prefix=/usr/local CFLAGS="$CFLAGS" CXXFLAGS="$CXXFLAGS" && \
    make -j$(nproc) V=0 && make install V=0

RUN --mount=type=cache,target=/ccache \
    git clone -b v3.12.1 https://github.com/Reference-LAPACK/lapack /opt/lapack-src && \
    mkdir /opt/lapack-build && cd /opt/lapack-build && \
    cmake -DCMAKE_INSTALL_LIBDIR=/opt/lapack -DBUILD_SHARED_LIBS=ON /opt/lapack-src && \
    cmake --build . -j$(nproc) --target install

RUN --mount=type=cache,target=/ccache \
    git clone -b dealii-9.7 https://github.com/dealii/dealii /opt/dealii-src && \
    mkdir /opt/dealii-build && cd /opt/dealii-build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
          -DCMAKE_C_COMPILER=/usr/bin/mpicc \
          -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx \
          -DCMAKE_Fortran_COMPILER=/usr/bin/mpif90 \
          -DDEAL_II_WITH_MPI=ON \
          -DDEAL_II_WITH_PETSC=ON \
          -DDEAL_II_WITH_P4EST=ON \
          -DDEAL_II_WITH_LAPACK=ON \
          -DPETSC_DIR=/opt/petsc \
          -DLAPACK_DIR=/opt/lapack \
          -DCMAKE_BUILD_TYPE=Release /opt/dealii-src && \
    make -j4 install
# Non usiamo nproc per deal.II per mancanza di ram. Sono povero :-\

FROM ubuntu:24.04
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/petsc /opt/petsc
COPY --from=builder /opt/lapack /opt/lapack
RUN apt-get update && apt-get install -y openmpi-bin openmpi-common libopenmpi-dev libtbb12 libboost-all-dev build-essential cmake git libtbb-dev pkg-config && rm -rf /var/lib/apt/lists/*
ENV PETSC_DIR=/opt/petsc
ENV LAPACK_DIR=/opt/lapack
ENV PKG_CONFIG_PATH=${PETSC_DIR}/lib/pkgconfig
