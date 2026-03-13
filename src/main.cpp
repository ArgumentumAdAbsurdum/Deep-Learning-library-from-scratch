
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
    c.add_layer(64, Activation::RELU);
    c.add_layer(64, Activation::RELU);
    c.add_layer(10,  Activation::SOFTMAX);
    c.configure_loss_function(Loss::CROSS_ENTROPY);
    
    c.initalise_random_weights(0,0.1);


    Dataset train = Dataset("../datasets/mnist_train.csv");
    train.normalize();
    train.one_hot_encode();
    
    Dataset test = Dataset("../datasets/mnist_test.csv");
    test.normalize();
    test.one_hot_encode();     

    //c.fit(200, train, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.001 , 256);

    ADAM_Optimizer adam;
    adam.lr = 0.001;
    adam.batch_size = 256;

    c.fit(50, train, adam);

    c.performance(train);
    c.performance(test);

    c.save_weights("test1.txt");
    
    
    
   
    /*
    Dataset ds;
    ds.input = Matrix::create_stacked_matrix(1,1,1024, 1);
    ds.expected = Matrix::create_stacked_matrix(1,1,1024, 1);

    NeuralNetwork c;

    c.configure_input_layer(1);
    c.add_layer(1, Activation::IDENTITY);
    c.configure_loss_function(Loss::QUADRATIC);
    c.initalise_random_weights();
    
    c.fit(1000, ds, Optimizer::MIN_BATCH_GRADIENT_DESCENT, 0.01, 1024);

    c.run(ds.input).print();
    */


    

    /*
    Matrix a = Matrix::create_stacked_matrix(10,1, 1, 2);
    Matrix b = Matrix::create_stacked_matrix(10,1, 1, 1);
    a.print();
    (activation<CUDA>::softmax(a)).print();
    */

}
