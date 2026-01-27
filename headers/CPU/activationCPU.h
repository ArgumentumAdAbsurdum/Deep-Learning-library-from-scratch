#pragma once
#include "activation.h"
#include "matrixCPU.h"

using activation_func = std::function<matrix<CPU>(const matrix<CPU>&)>;
using loss_func = std::function<float(const matrix<CPU>&, const matrix<CPU>& )>;
using loss_func_derivative = std::function<matrix<CPU>(const matrix<CPU>&, const matrix<CPU>&)>;


template<>
class activation<CPU>
{   
public:
    static matrix<CPU> identity(const matrix<CPU>& a);
    static matrix<CPU> ReLU(const matrix<CPU>& a);
    static matrix<CPU> ELU(const matrix<CPU>& a);
    static matrix<CPU> Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> Log_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> Hard_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> Tanh(const matrix<CPU>& a);
    static matrix<CPU> Softmax(const matrix<CPU>& a);


    static activation_func derivative_of(const activation_func& f);

private:
    static matrix<CPU> didentity(const matrix<CPU>& a);
    static matrix<CPU> dReLU(const matrix<CPU>& a);
    static matrix<CPU> dELU(const matrix<CPU>& a);
    static matrix<CPU> dSigmoid(const matrix<CPU>& a);
    static matrix<CPU> dLog_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> dHard_Sigmoid(const matrix<CPU>& a);
    static matrix<CPU> dTanh(const matrix<CPU>& a);

   inline static float ELU_ALPHA_PARAM = 1;

   
};



template<>
class loss<CPU>
{  
    public:
    static float Cross_Entropy(const matrix<CPU> &expected, const matrix<CPU> &result);
    static float Quadratic(const matrix<CPU> &expected, const matrix<CPU> &result);

    static loss_func_derivative derivative_of(const loss_func& f, const activation_func& afunc_output_layer);

    //private:
    static matrix<CPU> dCross_Entropy(const matrix<CPU> &expected, const matrix<CPU> &result);
    static matrix<CPU> dCross_Entropy_inkl_Softmax(const matrix<CPU> &expected, const matrix<CPU> &result);
    static matrix<CPU> dQuadratic(const matrix<CPU> &expected, const matrix<CPU> &result);


    
};
