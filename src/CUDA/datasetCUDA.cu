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
            this->input.push_back(matrix<CUDA>(input_row.size(), 1, input_row));
            this->expected.push_back(matrix<CUDA>(1,1, current_output)); 
        }
    }

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
}

dataset<CUDA>::dataset(const std::string filename, const std::vector<size_t>& ignore, size_t output_col)
{
    std::ifstream file(filename); 
    if (!file.is_open()) 
        throw std::runtime_error("dataset : Cannot open CSV file: " + filename); 

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
            this->input.push_back(matrix<CUDA>(input_row.size(), 1, input_row));
            this->expected.push_back(matrix<CUDA>(1,1, current_output)); 
        }
    }

    std::cout << "[LOADED " << filename << " SUCCESSFULLY ]" << std::endl; 
    
}

dataset<CUDA> dataset<CUDA>::split(float ratio)
{

    if(0 > ratio || ratio > 1)
        throw std::runtime_error("dataset : split argument needs to be between 0 and 1 ");
        
    size_t split_point = this->input.size() * ratio;

    std::vector<matrix<CUDA>> input_first_part(this->input.begin(), this->input.begin() + split_point);
    std::vector<matrix<CUDA>> input_second_part(this->input.begin() + split_point, this->input.end());

    std::vector<matrix<CUDA>> expected_first_part(this->expected.begin(), this->expected.begin() + split_point);
    std::vector<matrix<CUDA>> expected_second_part(this->expected.begin() + split_point, this->expected.end());

    this->input = input_first_part;
    this->expected = expected_first_part;

    dataset ds;
    ds.input = input_second_part;
    ds.expected = expected_second_part;
    return ds;

}

void dataset<CUDA>::one_hot_encode()
{
    if(this->expected[0].columns() != 1 || this->expected[0].rows() != 1)
        throw std::runtime_error("one_hot_encode : Wrong matrix output shape for one hot encoding. It needs to be 1x1xh."); 
    

    
    std::vector<matrix<CUDA>> res;
    res.reserve(this->expected.size());

    std::vector<float> values;
    values.reserve(this->expected.size());
    

    for(matrix<CUDA> &mat : this->expected)                            
        values.push_back(mat[0]);
    
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    for(matrix<CUDA> &mat : this->expected)
    {
        auto it = std::find(values.begin(), values.end(), mat[0]);
        int index = std::distance(values.begin(), it);

        matrix<CUDA> _x = matrix<CUDA>(values.size(), 1, 0);
        //_x[index] = 1.0;
        _x.set(index, 1);

        res.push_back(_x);
    }

    this->expected = res;
    
}



void dataset<CUDA>::normalize()
{
    float max = std::numeric_limits<float>::min();
    float min = std::numeric_limits<float>::max();

    for(matrix<CUDA> vec : this->input )
    {
        float current_min = vec.min()[0];
        float current_max = vec.max()[0];         

        max = std::max(current_max, max);
        min = std::min(current_min, min);
    }

    if(max == min)
        throw std::runtime_error("normalize : All values of the dataset are the same, which results in a divison by zero.");

    for(matrix<CUDA>& vec : this->input)
    {
        
        vec = vec - min;
        vec = vec * (1 / (max - min));
    }
    
    
}


void dataset<CUDA>::standardize()
{
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

}

void dataset<CUDA>::print_information()
{

    if(this->input.empty())
    {
        std::cout << "Dataset is empty." << std::endl; 
    }
    
    std::cout << "Samples : " << this->input.size() << std::endl;
    std::cout << "Input vector dim : " << this->input[0].rows() << std::endl;
    std::cout << "Output vector dim : " << this->expected[0].rows() << std::endl;

}
