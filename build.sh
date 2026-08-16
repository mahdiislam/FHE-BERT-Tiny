#!/bin/zsh
set -e
echo "Removing previous build folder"
rm -rf build/
mkdir build && cd build
echo $PWD
echo "Starting new build"
cmake -DCMAKE_PREFIX_PATH=$HOME/openfhe-install ..
make -j8
