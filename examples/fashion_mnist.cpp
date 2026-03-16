#include <iostream>
#include "DeepModel.h"


const std::string path = "datasets/fashion-mnist_train.csv";

int main()
{

    Dataset data = Dataset(path);
    data.normalize();
    data.one_hot_encode();

    auto [train, test] = data.split(0.8);
    test.print_information();

    NeuralNetwork nn;

    nn.configure_input_layer(784);
    nn.add_layer(256, Activation::RELU);
    nn.add_layer(128, Activation::RELU);
    nn.add_layer(10,  Activation::SOFTMAX);
    nn.configure_loss_function(Loss::CROSS_ENTROPY);
    
    nn.initalise_he_weights();
    
    ADAM_Optimizer adam;
    adam.lr = 0.001;
    adam.batch_size = 64;

    nn.fit(30, train, adam);

    nn.performance(test);



}