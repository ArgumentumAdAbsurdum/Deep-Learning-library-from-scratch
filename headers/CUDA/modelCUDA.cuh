#pragma once
#include "DeepModel.h"
#include "activationCUDA.cuh"
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>



template<>
class dataset<CUDA>
{

public:
    matrix<CUDA> input;
    matrix<CUDA> expected;
    
    dataset();
    dataset(const std::string filename, size_t label_col = 0);
    dataset(const std::string filename, const std::vector<size_t>& ignore, size_t label_col = 0);

    dataset split(float ratio);


    void one_hot_encode();   
    void normalize ();
    void standardize();


    void print_information();

};



template<>
class neuralnetwork<CUDA> 
{   
private:

    bool imported = false;
    loss<CUDA> loss_function_class;

    size_t lfunc_type;
    std::vector<size_t> afunc_type;


    size_t input_layer_neurons;
    size_t output_layer_neurons;

    loss_fn lfunc;
    loss_derivative_fn lfunc_dx;

    std::vector<size_t> neurons_per_layer;
    std::vector<activation_fn> afunc;
    std::vector<activation_fn> afunc_dx;

    std::vector<matrix<CUDA>> weight_matrices;
    std::vector<matrix<CUDA>> bias_matrices;


    void gradient_descent(const size_t steps, dataset<CUDA> &ds, double lr, double lambda, size_t batch_size);
    
    std::vector<matrix<CUDA>> layer_outputs(const matrix<CUDA>& input);
    

    

public:

    neuralnetwork<CUDA>();

    void add_layer(const size_t neurons, activation_type  atype);
    void configure_loss_function(loss_type ltype);
    void set_loss_weights(const std::vector<float> w);
    void configure_input_layer(const size_t neurons);


    void initalise_random_weights(float begin = -0.1, float end = 0.1);
    void initalise_xavier_weights();
    void initalise_he_weights();

    void fit(const size_t epochs,dataset<CUDA> &ds, optimizer_type ofunc, double lr, size_t batch_size = 64);
    void fit(const size_t epochs,dataset<CUDA> &ds, optimizer_type ofunc, hyperparameter<CUDA> &param);
    void fit(const size_t epochs,dataset<CUDA> &ds, adam_optimizer<CUDA> &adam);

    matrix<CUDA> run(const matrix<CUDA>& input);

    void performance(dataset<CUDA>& ds, std::string name);
    void performance(dataset<CUDA>& ds);
    
    void binary_confusion_matrix(dataset<CUDA>& ds, const float threshold = 0.5);

    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename);
};
