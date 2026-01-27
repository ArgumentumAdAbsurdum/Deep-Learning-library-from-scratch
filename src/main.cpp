#include <iostream>
#include "modelCPU.h"
#include <chrono>
#include <experimental/simd>
#include <omp.h>

int main()
{
    Classificator c;

    c.add_layer(16, Activation::ReLU);
    c.add_layer(16, Activation::ReLU);
    c.add_layer(10, Activation::Softmax);
    c.configure_loss_function(Loss::Cross_Entropy);
    c.load_csv("../datasets/mnist_train.csv");

    c.initalise();

    c.fit(1);


}