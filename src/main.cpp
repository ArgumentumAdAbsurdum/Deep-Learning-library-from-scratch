
#define ENABLE_CUDA  1
#include <iostream>
#include "DeepModel.h"
#include "activationCUDA.cuh"
#include <chrono>
#include <experimental/simd>
#include <omp.h>




int main()
{
    // cmake .. -DENABLE_CUDA=ON 
    
    
    NeuralNetwork c;

    c.configure_input_layer(784);
    c.add_layer(256, Activation::RELU);
    c.add_layer(128, Activation::RELU);
    c.add_layer(10, Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise_random_weights();

    
    Dataset train = Dataset("../datasets/mnist_train.csv");
    Dataset test = Dataset("../datasets/mnist_test.csv");
 
    train.normalize();
    test.normalize();

    train.one_hot_encode();
    test.one_hot_encode();
    
    c.fit(1, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.001 , 1);

    c.performance(train);
    c.performance(test);

    c.save_weights("test1.txt");
    

    // softmax, dsoft_inkl_cross, relu , drelu, transpose, reciprocal, +, -, %, scale

    
    /*
    Matrix a(2,1, 1);
    Matrix b(2,1, 0);

    NeuralNetwork c;
    c.configure_input_layer(2);
    c.add_layer(2, Activation::RELU);
    c.configure_loss_function(Loss::QUADRATIC);

    c.initalise_random_weights(0,1);

    Dataset d;
    d.input = {a,b};
    d.expected = {a,b};



    c.fit(1000, d, Optimizer::STOCHASTIC_GRADIENT_DESCENT, 0.01);

    c.run(a).print();
    */

}
