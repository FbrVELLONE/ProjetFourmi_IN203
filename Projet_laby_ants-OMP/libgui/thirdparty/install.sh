#!/bin/env sh
#module load gcc/5.2
tar xzvf freeglut-3.0.0.tar.gz
mkdir -p build
cd build
cmake ../freeglut-3.0.0 -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/../../../
make all install
