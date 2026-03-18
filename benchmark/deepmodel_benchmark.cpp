#include <iostream>
#include "DeepModel.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

const std::string path = "datasets/mnist_test.csv";


void benchmark(const size_t batch_size, const size_t epochs, NeuralNetwork nn, Dataset &train, Dataset& test)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    nn.fit(epochs, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.001 , batch_size);
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = end - start;
    std::cout << "[batch_size = " << batch_size << ", epochs = " << epochs << " => time : " << time.count() << "s , accuracy : "  << nn.accuracy(test) * 100 << "% ]" << std::endl;
  
    
}

void benchmark_adam(const size_t batch_size, const size_t epochs, NeuralNetwork nn, Dataset &train, Dataset& test)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    ADAM_Optimizer adam;
    adam.lr = 0.001;
    adam.batch_size = batch_size;
    adam.lambda = 10e-4;

    nn.fit(epochs, train, adam);
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time = end - start;
    std::cout << "[batch_size = " << batch_size << ", epochs = " << epochs << " => time : " << time.count() << "s , accuracy : "  << nn.accuracy(test) * 100 << "% ]" << std::endl;
  
    
}

int main()
{
    omp_set_num_threads(8);
    Dataset data = Dataset(path);
    
    data.normalize();
    data.one_hot_encode();     

    auto [train, test] = data.split(0.8);

    NeuralNetwork nn;
    nn.disable_print();

    nn.configure_input_layer(784);
    nn.add_layer(128, Activation::RELU);
    nn.add_layer(128, Activation::RELU);
    nn.add_layer(10,  Activation::SOFTMAX);
    nn.configure_loss_function(Loss::CROSS_ENTROPY);
    
    nn.initalise_he_weights();
    
    std::cout << "====[Benchmark for MNIST dataset with 60k samples]====" << std::endl;
    std::cout << "  --> Neuralnetwork: 784 x 128 x 128 x10" << std::endl;
    std::cout << "  --> Activation functions : ReLU ReLU Softmax" << std::endl;
    std::cout << "  --> Loss function : Cross Entropy" << std::endl;
    std::cout << "  -->learnrate : 0.001 (for both runs)" << std::endl;


    std::cout << "====[Mini Batch Gradient descent:]=====================" << std::endl;
    benchmark(1, 1, nn, train, test);
    benchmark(2, 1, nn, train, test);
    benchmark(4, 1, nn, train, test);
    benchmark(8, 1, nn, train, test);
    benchmark(16, 20, nn, train, test);
    benchmark(32, 20, nn, train, test);
    benchmark(64, 20, nn, train, test);


    std::cout << "====[Mini batch gradient descent with Adam and L2 regulazation enabled]====" << std::endl;
    std::cout << "  --> beta1 = 0.9, beta2 = 0.999, epsilon = 10e-8, lambda = 10e-4" << std::endl;
    benchmark_adam(1, 1, nn, train, test);
    benchmark_adam(2, 1, nn, train, test);
    benchmark_adam(4, 1, nn, train, test);
    benchmark_adam(8, 1, nn, train, test);
    benchmark_adam(16, 20, nn, train, test);
    benchmark_adam(32, 20, nn, train, test);
    benchmark_adam(64, 20, nn, train, test);




}
