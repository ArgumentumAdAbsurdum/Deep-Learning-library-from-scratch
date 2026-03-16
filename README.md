




cmake -B build -DENABLE_CUDA=ON -DBUILD_BENCHMARK=ON
make --build build
./build/benchmark/deepmodel_benchmark

cmake -B build -DENABLE_CUDA=ON -DBUILD_EXAMPLES=ON
cmake --build build --target mnist_example