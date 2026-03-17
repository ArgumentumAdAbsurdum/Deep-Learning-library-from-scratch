
/**
 * @file fashion_mnist.cpp
 * @example Demonstrates matrix operations with stacked matrices.
 */

#include <iostream>
#include "DeepModel.h"


int main()
{

    // create a stacked matrix of shape 5x1x100 with random values from 0 to 2.
    Matrix data = Matrix::create_stacked_matrix(5,1,100, 0, 2.0f);


    const float n = (float)data.height();
    
    // Calculate the mean by adding all stacked matrices together and dividing by height.
    Matrix mean = Matrix::reduce_sum(data) * (1/n) ;    // shape 5x1x1

    // Calculate the variance by squaring all elemenets and then adding the square sums toghether.
    Matrix variance = Matrix::reduce_sum(Matrix::square(data)) * (1/n); // shape 5x1x1

    // Sqrt all values to geht the variance
    Matrix standard_deviation = Matrix::sqrt(variance);     


    // Standardize the data, by using the intern broadcasting definitions:
    // data -> shape 5x1x100
    // mean -> shape 5x1x1
    // standard_deviation -> shape 5x1x1
    // => the substraction and the hadamard procduct will be broadcasted to all values.

    Matrix standardized_data = (data - mean) % Matrix::reciprocal(standard_deviation); 


    std::cout << "[MEAN of our data below]" << std::endl;
    mean.print();
    std::cout << "----------------------------------" << std::endl;

    std::cout << "[STANDARD DEVIATION of our data below]" << std::endl;
    standard_deviation.print();
    std::cout << "-----------------------------------" << std::endl;
}
