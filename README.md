# DeepModel - A C++ Deep Learning Libary
> A high performance neural network libary written from scratch in C++, with optional CUDA support.

---
## Overview
This libary is entirely in C++ implemented and has a custom linear algebra engine, which is optimized for CUDA and CPU-only execution.\n

Every operation like matrix transposing, matrix multiplication and the entire optimization algorithm is implemented by hand.
It also contains a own Dataset class, which gives the user the ability to interpret and edit .csv datasets.

The Github repo contains training examples with mnist and fashion-mnist, while being also benchmarked against pytorch.

---
## Features


### Core Features
- **Backpropagation with L2 regularization**
- **Weighted loss support**
- **Random / Xavier / He weight initalization**
- **ADAM and ADAMW**
- **Dataset editing**


### Optimizers
`ADAM_OPTIMIZER` `STOCHASTIC GRADIENT DESCENT` `BATCH GRADIENT DESCENT` `MINI BATCH GRADIENT DESCENT`

### Activation functions
`RELU`  `IDENTITY`  `ELU`  `SIGMOID`  `LOG_SIGMOID`  `HARD_SIGMOID`  `TANH`  `SOFTMAX`

### Loss functions
`CROSS ENTROPY`  `QUADRATIC (MLE)`


---
## How to build with cmake

### Requirements
C++17 GNU / Clang 
OpenMP  
CUDA Toolkit (Optional for CUDA version)
Cmake


### CPU-only

```bash
mkdir build
cmake -B build -DENABLE_CUDA=OFF
cmake --build build
```


### CUDA Support

```bash
mkdir build
cmake -B build -DENABLE_CUDA=ON
cmake --build build
```

### Adding your own files

Add this to the 'CMakeLists.txt':
```cmake
add_executable(my_programm my_programm.cpp)
target_link_libaries(my_program PRIVATE DeepModel)
```

Then run: 

```bash
cmake --build build
./build/my_program
```

--
## Quick start



cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
make --build build
./build/benchmark/deepmodel_benchmark

cmake -B build -DENABLE_CUDA=ON -DBUILD_EXAMPLES=ON
cmake --build build --target mnist_example