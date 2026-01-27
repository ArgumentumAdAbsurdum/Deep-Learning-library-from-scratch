#pragma once
#include "model.h"
#include "matrix.h"
#include "activationCPU.h"
#include <string>


class model
{
protected:
    
    size_t input_layer_neurons;
    size_t output_layer_neurons;

    loss_func lfunc;
    loss_func_derivative lfdx;

    std::vector<size_t> neurons_per_layer;
    std::vector<activation_func> afuncs;
    std::vector<activation_func> afdx;


    model();

public:

    void add_layer(const size_t neurons, activation_func  afunc);
    void configure_loss_function(loss_func lfunc);


};

template<>
class classificator<CPU> : public model
{   
private:

    std::vector<matrix<CPU>> weight_matrices;

public:

    std::vector<matrix<CPU>> input;
    std::vector<matrix<CPU>> expected;

    classificator<CPU>();
    void load_csv(const std::string& filename, size_t label_col = 0);

    void initalise();
    void fit(size_t epochs);
    std::vector<matrix<CPU>> layer_outputs(const matrix<CPU>& input);
};
