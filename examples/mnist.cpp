/**
 * @file mnist.cpp
 * @example Demonstrates training a neural network on the mnist dataset with ADAM.
 * @note place the .csv in datasets or edit the path.
 */



#include <iostream>
#include "DeepModel.h"
#include <filesystem>

// Path to the mnist dataset as .csv
const std::string path = "datasets/mnist_train.csv";
 



int main()
{

    if(!std::filesystem::exists(path))
    {
        std::cerr << "Error : Dataset not found at: " << path << " , please edit path or download & place the mnist_train.csv inside /datasets." << std::endl; 
    }

    // Load dataset and edit it
    Dataset data = Dataset(path);
    data.normalize();
    data.one_hot_encode();

    // split the dataset and print information
    auto [train, test] = data.split(0.8);
    test.print_information();


    // Create a new Network
    NeuralNetwork nn;

    nn.configure_input_layer(784);
    nn.add_layer(64, Activation::RELU);
    nn.add_layer(64, Activation::RELU);
    nn.add_layer(10,  Activation::SOFTMAX);
    nn.configure_loss_function(Loss::CROSS_ENTROPY);
    
    // initalise weights
    nn.initalise_he_weights();
    
    // Configure ADAM
    ADAM_Optimizer adam;
    adam.lr = 0.001;
    adam.batch_size = 64;

    // Run the Backpropagation
    nn.fit(30, train, adam);

    // Print accuracy
    nn.performance(test);

    nn.save_weights("mnist_example_weights.txt");


}
