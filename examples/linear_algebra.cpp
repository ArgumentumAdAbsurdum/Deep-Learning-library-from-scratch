#include <iostream>
#include "DeepModel.h"


int main()
{
    Matrix data = Matrix::create_stacked_matrix(5,1,100, 0, 2.0f);

    const float n = (float)data.height();
    Matrix mean = Matrix::reduce_sum(data) * (1/n) ;
    Matrix variance = Matrix::reduce_sum(Matrix::square(data)) * (1/n);
    Matrix standard_deviation = Matrix::sqrt(variance);

    Matrix standardized_data = (data - mean) % Matrix::reciprocal(standard_deviation); 


    std::cout << "[MEAN of our data below]" << std::endl;
    mean.print();
    std::cout << "----------------------------------" << std::endl;

    std::cout << "[STANDARD DEVIATION of our data below]" << std::endl;
    standard_deviation.print();
    std::cout << "-----------------------------------" << std::endl;
}
