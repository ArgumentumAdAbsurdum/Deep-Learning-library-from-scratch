#include "modelCUDA.cuh"
#include <algorithm>
#include <fstream>    
#include <sstream>    
#include <string>    
#include <vector>     
#include <stdexcept>  
#include <cmath>

dataset<CUDA>::dataset()
{
    
}

dataset<CUDA>::dataset(const std::string filename, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 


    std::vector<float> all_input_values;
    std::vector<float> all_expected_values;
    size_t rows = 0;
    size_t height = 0;

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        float current_output = -1; 
        size_t col = 0; 
        bool skip_row = false;

        while (std::getline(ss, cell, ',')) 
        { 
            try
            {
                float value = std::stof(cell);
                if (std::isnan(value)) {
                    skip_row = true;
                    break;
                }


                if (col == output_col) 
                    current_output = static_cast<float>(value); 
                else 
                    input_row.push_back(value); 
                ++col; 
            }
            catch(...)
            {
                skip_row = true;
                break;
            }

        }

        if(!input_row.empty() && !skip_row)
        {
            all_input_values.insert(all_input_values.end(), input_row.begin(), input_row.end());
            all_expected_values.push_back(current_output);
            rows = input_row.size();
            height++;
        }
    }

    this->input = matrix<CUDA>::create_stacked_matrix(rows, 1, height, all_input_values);
    this->expected = matrix<CUDA>::create_stacked_matrix(1, 1, height, all_expected_values);

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
}

dataset<CUDA>::dataset(const std::string filename, const std::vector<size_t>& ignore, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 


    std::vector<float> all_input_values;
    std::vector<float> all_expected_values;
    size_t rows = 0;
    size_t height = 0;

    std::string line;
    while (std::getline(file, line)) 
    { 
        std::stringstream ss(line); 
        std::string cell; 
        std::vector<float> input_row;
        float current_output = -1; 
        size_t col = 0; 
        bool skip_row = false;

        while (std::getline(ss, cell, ',')) 
        { 
            try
            {

                if (std::find(ignore.begin(), ignore.end(), col) != ignore.end()) 
                {   
                    ++col;
                    continue;
                }

                float value = std::stof(cell); 
                if (std::isnan(value)) {
                    skip_row = true;
                    break;
                }

                if (col == output_col) 
                    current_output = static_cast<float>(value); 
                else 
                    input_row.push_back(value); 
                ++col; 
            }
            catch(...)
            {
                skip_row = true;
                break;
            }

        }

        if(!input_row.empty() && !skip_row)
        {
            all_input_values.insert(all_input_values.end(), input_row.begin(), input_row.end());
            all_expected_values.push_back(current_output);
            rows = input_row.size();
            height++;
        }
    }

    this->input = matrix<CUDA>::create_stacked_matrix(rows, 1, height, all_input_values);
    this->expected = matrix<CUDA>::create_stacked_matrix(1, 1, height, all_expected_values);
    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
    
}

dataset<CUDA> dataset<CUDA>::split(float ratio)
{


}

void dataset<CUDA>::one_hot_encode()
{
    if(this->expected.columns() != 1 || this->expected.rows() != 1)
        throw std::runtime_error("one_hot_encode : Wrong matrix output shape for one hot encoding. It needs to be 1x1xh."); 
    

    
    std::vector<float> all_new_expected_values;


    std::vector<float> values = expected.values();

    std::vector<float> unique_values = expected.values();
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());


    for(size_t i = 0; i < this->expected.height();  i++)
    {   
        
        auto it = std::find(unique_values.begin(), unique_values.end(), values[i]);
        int index = std::distance(unique_values.begin(), it);

        std::vector<float> _x(unique_values.size(), 0);
        _x[index] = 1.0;
        all_new_expected_values.insert(all_new_expected_values.end(), _x.begin(), _x.end());
    }

   this->expected = matrix<CUDA>::create_stacked_matrix(unique_values.size(), 1 , this->expected.height(), all_new_expected_values);
    
}



void dataset<CUDA>::normalize()
{

    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    std::vector<float> layer_max = this->input.max();
    std::vector<float> layer_min = this->input.min();

    for(int i = 0; i < this->input.height(); i++)
    {
        max = std::max(layer_max[i], max);
        min = std::min(layer_min[i], min);
    }

    if(max == min)
        throw std::runtime_error("normalize : All values of the dataset are the same, which results in a divison by zero.");

    this->input = (this->input - min) * (1 / (max - min));

}


void dataset<CUDA>::standardize()
{
    /*
    size_t rows = this->input[0].rows();
    size_t n = this->input.size();

    matrix<CUDA> mean(rows, 1);
    for(matrix<CUDA> &vec : this->input)
        mean = mean + vec;
    mean = mean * (1 / (float) n);

    matrix<CUDA> variance(rows, 1);
    for(matrix<CUDA> &vec : this->input)
    {
        matrix<CUDA> diff = vec - mean;
        variance = variance + matrix<CUDA>::square(diff);
    }

    variance = variance * (1 / (float) n);
    matrix<CUDA> sigma = matrix<CUDA>::sqrt(variance);

    for(matrix<CUDA>& vec : this->input)
        vec = (vec - mean) % matrix<CUDA>::reciprocal(sigma);
    */
}

void dataset<CUDA>::print_information()
{

    if(this->input.empty())
    {
        std::cout << "Dataset is empty." << std::endl; 
    }
    
    std::cout << "Samples : " << this->input.height() << std::endl;
    std::cout << "Input vector dim : " << this->input.rows() << std::endl;
    std::cout << "Output vector dim : " << this->expected.rows() << std::endl;

}
