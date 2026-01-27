#include "modelCPU.h"
#include "activationCPU.h"
#include <fstream>
#include <sstream>
#include <stdexcept>


model::model()
{

}

void model::add_layer(const size_t neurons, activation_func afunc) 
{
    neurons_per_layer.push_back(neurons);
    afuncs.push_back(afunc);

    this->output_layer_neurons = neurons;
}


void model::configure_loss_function(loss_func lfunc)
{
    this->lfunc = lfunc;
}

classificator<CPU>::classificator() : model()
{}

void classificator<CPU>::load_csv(const std::string &filename, size_t label_col ) 
{
    if (output_layer_neurons == 0)
        throw std::runtime_error("Output layer not configured. Run configure_output_layer(neurons, activation func) first.");

    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open CSV file: " + filename);

    input.clear();
    expected.clear();

    std::string line;
    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;


        std::vector<float> input_row;

        float current_label = -1;

        

        size_t col = 0;
        while (std::getline(ss, cell, ','))
        {
        
            float value = std::stof(cell);
            
            if (col == label_col)
                current_label = value;
            else
                input_row.push_back(value);
            ++col;
        }
        
        
        if(input_row.size() - 1 > this->input_layer_neurons)
            this->input_layer_neurons = input_row.size(); // because of the bias
        
        
        
        matrix<CPU> x_(
                        {input_row.size(), 1}, 
                        input_row
        );
        

        input.push_back(x_);
        
            

        if (current_label < 0 || current_label >= static_cast<int>(output_layer_neurons))
            throw std::runtime_error("Label out of range: " + std::to_string(current_label));

          
        std::vector<float> one_hot(output_layer_neurons, 0.0f);
        one_hot[static_cast<int>(current_label)] = 1.0f;

        
        matrix<CPU> y_(
                        {one_hot.size(), 1}, 
                        one_hot
        );
        
        expected.push_back(y_);

    }

    neurons_per_layer.insert(neurons_per_layer.begin(), this->input_layer_neurons);

    if (input_layer_neurons == 0 && !input.empty())
        input_layer_neurons = input.front().size();


    std::cout << "[LOADED " + filename + " SUCCESSFULLY]" << std::endl;
}

void classificator<CPU>::initalise()
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i] + 1;      
        std::cout << "[LAYER = " << i << "=> ROWS = " << rows << " , COLUMNS = " << cols << " ] " << std::endl;

        matrix<CPU> mat({rows, cols}, 0, 1);       
        weight_matrices.push_back(mat);
    }

    for(activation_func afunc : afuncs)
        afdx.push_back(activation<CPU>::derivative_of(afunc));
    
    lfdx = loss<CPU>::derivative_of(this->lfunc, afuncs.back());
    
}

std::vector<matrix<CPU>> classificator<CPU>::layer_outputs(const matrix<CPU> &input)
{
    std::vector<matrix<CPU>> outputs;
    outputs.resize(neurons_per_layer.size() + 1);

    std::cout << outputs[0].raw() << std::endl;
    outputs[0] = input;

    matrix<CPU> current = outputs[0];
    

    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        
        
        current.insert_row(0, 1);
        //std::cout << weight_matrices[i].rows() << " " << weight_matrices[i].columns() << std::endl;
        //std::cout << current.rows() << " " << current.columns() << std::endl;

        matrix<CPU> prod = weight_matrices[i] * current;
        current = afuncs[i](prod);

        outputs[i+1] = prod;
    }

    return outputs;
}

void classificator<CPU>::fit(size_t epochs)
{
    
   
    std::vector<matrix<CPU>> Z = layer_outputs(input[0]);
    size_t index = neurons_per_layer.size() -1;

    
    matrix<CPU> Zb = Z[index - 1];
    Zb.insert_row(0, 1);

    matrix<CPU> delta_index = lfdx(Z[0], input[0]) % afdx[index](weight_matrices[index] * Zb);

    for(; index >= 1; index--)
    {
        Zb = Z[index - 2];
        Zb.insert_row(0, 1);
        delta_index = (weight_matrices[index] * delta_index) % (afdx[index -1](weight_matrices[index-1] * Z[index - 2]));
    }

    
}


