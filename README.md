# DeepModel - A C++ Deep Learning Libary
> A high performance neural network libary written from scratch in C++, with optional CUDA support.

---
## Overview
This libary is entirely in C++ implemented and has a custom linear algebra engine, which is optimized for CUDA and CPU-only execution.
Every operation like matrix transposing, matrix multiplication and the entire optimization algorithm is implemented by hand. 
The Github repo contains training examples with mnist and fashion-mnist, while being also benchmarked against pytorch.

---
## Features


cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
make --build build
./build/benchmark/deepmodel_benchmark

cmake -B build -DENABLE_CUDA=ON -DBUILD_EXAMPLES=ON
cmake --build build --target mnist_example