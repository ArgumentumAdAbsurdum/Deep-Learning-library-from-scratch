#include "modelCPU.h"





neuralnetwork<CPU>::neuralnetwork() : input_layer_neurons(0)
{}

void neuralnetwork<CPU>::add_layer(const size_t neurons, activation_type atype) 
{
    neurons_per_layer.push_back(neurons);
    afunc_type.push_back(atype);
    this->output_layer_neurons = neurons;
}

void neuralnetwork<CPU>::configure_loss_function(loss_type _ltype) 
{
    this->lfunc_type = _ltype;
}

void neuralnetwork<CPU>::configure_input_layer(const size_t neurons)
{
    this->input_layer_neurons = neurons;
    neurons_per_layer.insert(neurons_per_layer.begin(), neurons);
}





// ------------------------------- BACKPROPAGATION -------------------------------------------

void neuralnetwork<CPU>::initalise()
{
    std::cout << "[INITALISE]" << std::endl;
    if(neurons_per_layer.size() <= 1)
        throw std::runtime_error("Initalise() : run add_layer first to build the NN. ");

    for(int i = 0; i < neurons_per_layer.size()-1; i++)
    {
        size_t rows = neurons_per_layer[i+1];        
        size_t cols = neurons_per_layer[i];      

        std::cout << "[LAYER = " << i << " WEIGHT MATRIX: => ROWS = " << rows << " , COLUMNS = " << cols << ", BIAS = " <<  rows << "] " << std::endl;

        matrix<CPU> mat(rows, cols, -0.1, 0.1);    
        weight_matrices.push_back(mat);

        matrix<CPU> bias(rows , 1, -0.1, 0.1);
        bias_matrices.push_back(bias);
    }



    lfunc = loss<CPU>::get_fn(lfunc_type);
    lfunc_dx = loss<CPU>::get_derivative_fn(lfunc_type, afunc_type.back());

    for(size_t a : afunc_type)
    {
        afunc.push_back(activation<CPU>::get_fn(a));
        afunc_dx.push_back(activation<CPU>::get_derivative_fn(a));
    }


    if(afunc_type.back() == activation<CPU>::SOFTMAX && this->lfunc_type != loss<CPU>::CROSS_ENTROPY )
        throw std::invalid_argument("Activation function softmax only works with cross entropy loss function.");

}


matrix<CPU> neuralnetwork<CPU>::run(const matrix<CPU> &input)
{
    matrix<CPU> result = input;
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {

        matrix<CPU> prod = weight_matrices[i] * result + bias_matrices[i];
        
        result = afunc[i](prod);
    }

    return result;
}



std::vector<matrix<CPU>> neuralnetwork<CPU>::layer_outputs(const matrix<CPU> &input)
{
    std::vector<matrix<CPU>> outputs;
    outputs.resize(neurons_per_layer.size());
    outputs[0] = input;

    matrix<CPU> current = outputs[0];
    
    for(int i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        matrix<CPU> prod = weight_matrices[i] * current + bias_matrices[i];
        
        current = afunc[i](prod);

        outputs[i+1] = current;
    }
    return outputs;
}


void neuralnetwork<CPU>::gradient_descent(const size_t epochs, dataset<CPU>& ds, double lr , size_t batch_size)
{
    std::vector<matrix<CPU>> wgradients;
    wgradients.resize(weight_matrices.size());

    std::vector<matrix<CPU>> bgradients;
    bgradients.resize(bias_matrices.size());

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ds.input.size() / batch_size - 1); 


    for(size_t i = 0; i < weight_matrices.size(); i++)
    {
        wgradients[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        bgradients[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);
    }

    for(size_t ep = 0; ep < epochs; ep++)
    {

        // ----------------------------------
        std::cout << ep << std::endl;
        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            wgradients[i].set(0);
            bgradients[i].set(0);
        }

        size_t start = dist(rng);
        std::vector<matrix<CPU>> Z;
        Z.reserve(this->weight_matrices.size() + 1);

        for(size_t pos = start; pos < start + batch_size; pos++)
        {
            matrix<CPU> &input_value = ds.input[pos];
            matrix<CPU> &truth_value = ds.expected[pos];
            Z = layer_outputs(input_value);


            ssize_t index = Z.size() - 1;
            matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]); 

            wgradients[index-1] += delta * Z[index-1].transpose();
            bgradients[index-1] += delta;
            
            index-= 2;
            for(index; index >= 0; index--)  
            {    
                matrix<CPU> weight_transposed = weight_matrices[index+1].transpose();
        
                delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index] + bias_matrices[index]);
                
                wgradients[index] += delta * Z[index].transpose();
                bgradients[index] += delta;
            }
            
        }

        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            weight_matrices[i] -= wgradients[i] * (lr / ((float) batch_size)); 
            bias_matrices[i] -= bgradients[i] * (lr / ((float) batch_size));
        }

    }
}




/*
void neuralnetwork<CPU>::stochastic_gradient_descent(const size_t epochs, dataset<CPU>& ds, double lr)
{

    
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,ds.input.size()-1); 

    for(size_t ep = 0; ep < epochs; ep++)
    {   

        size_t pos = dist(rng);
        matrix<CPU> &input_value = ds.input[pos];
        matrix<CPU> truth_value = ds.expected[pos];

        std::vector<matrix<CPU>> Z = layer_outputs(input_value);

        
        ssize_t index = Z.size() - 1;
        matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]); 

        
        matrix<CPU> wgradient = delta * Z[index-1].transpose();
        matrix<CPU>& bgradient = delta;
        

        weight_matrices[index-1] -= lr * wgradient;
        bias_matrices[index-1] -= lr * bgradient;
        
        

        index -=2;
        for(index; index >= 0; index--)  
        {    
            matrix<CPU> weight_transposed = weight_matrices[index+1].transpose();
        
            delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index]);
            wgradient = delta * Z[index].transpose();
            bgradient = delta;

            weight_matrices[index] -= lr * wgradient;
            bias_matrices[index] -= lr * bgradient;
        }

    }
    
}
*/








// ------------------------------------------------------------------------------------

void neuralnetwork<CPU>::fit(const size_t epochs, dataset<CPU>& ds, optimizer_type ofunc, double lr, size_t batch_size )
{

    switch(ofunc)
    {
        case optimizer<CPU>::STOCHASTIC_GRADIENT_DESCENT:
            gradient_descent(epochs, ds, lr, 1);
            break;

        case optimizer<CPU>::BATCH_GRADIENT_DESCENT:
            gradient_descent(epochs, ds, lr, ds.input.size());
            break;

        case optimizer<CPU>::MIN_BATCH_GRADIENT_DESCENT:
            gradient_descent(epochs, ds, lr, batch_size);
            break;
    }
}

void neuralnetwork<CPU>::fit(const size_t epochs, dataset<CPU> &ds, optimizer_type ofunc, adam_optimizer<CPU> &adam, size_t batch_size)
{
    switch(ofunc)
    {
        case optimizer<CPU>::STOCHASTIC_GRADIENT_DESCENT:
            batch_size = 1;
            break;

        case optimizer<CPU>::BATCH_GRADIENT_DESCENT:
            batch_size = ds.input.size();
            break;

        case optimizer<CPU>::MIN_BATCH_GRADIENT_DESCENT:
            break;
    }

    std::vector<matrix<CPU>> weight_gradients;
    weight_gradients.resize(weight_matrices.size());

    std::vector<matrix<CPU>> bias_gradients;
    bias_gradients.resize(bias_matrices.size());


    std::vector<matrix<CPU>> weight_momentum;
    weight_momentum.resize(weight_matrices.size());

    std::vector<matrix<CPU>> bias_momentum;
    bias_momentum.resize(weight_matrices.size());

    std::vector<matrix<CPU>> weight_variance;
    weight_variance.resize(weight_matrices.size());
        
    std::vector<matrix<CPU>> bias_variance;
    bias_variance.resize(weight_matrices.size());



    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, ds.input.size() / batch_size - 1); 


    for(size_t i = 0; i < weight_matrices.size(); i++)
    {
        weight_gradients[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        bias_gradients[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);

        weight_momentum[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);
        weight_variance[i] = matrix<CPU>(weight_matrices[i].rows(), weight_matrices[i].columns(), 0);

        bias_momentum[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);
        bias_variance[i] = matrix<CPU>(bias_matrices[i].rows(), 1, 0);
    }

    for(size_t ep = 1; ep <= epochs; ep++)
    {

        for(size_t i = 0; i < weight_matrices.size(); i++)
        {
            weight_gradients[i].set(0);
            bias_gradients[i].set(0);
        }

        size_t start = dist(rng);
        std::vector<matrix<CPU>> Z;
        Z.reserve(this->weight_matrices.size() + 1);

        for(size_t pos = start; pos < start + batch_size; pos++)
        {
            matrix<CPU> &input_value = ds.input[pos];
            matrix<CPU> &truth_value = ds.expected[pos];
            Z = layer_outputs(input_value);


            ssize_t index = Z.size() - 1;
            matrix<CPU> delta = lfunc_dx(Z[index], truth_value) % afunc_dx[index-1](weight_matrices[index-1] * Z[index-1] + bias_matrices[index-1]); 

            weight_gradients[index-1] += delta * Z[index-1].transpose();
            bias_gradients[index-1] += delta;
            
            index-= 2;
            for(index; index >= 0; index--)  
            {    
                matrix<CPU> weight_transposed = weight_matrices[index+1].transpose();
        
                delta = (weight_transposed * delta) % afunc_dx[index](weight_matrices[index] * Z[index] + bias_matrices[index]);
                
                weight_gradients[index] += delta * Z[index].transpose();
                bias_gradients[index] += delta;
            }
            
        }

        for(size_t i = 0; i < weight_matrices.size(); i++)
        {

            weight_gradients[i] = weight_gradients[i] * (1 / ((float) batch_size));
            bias_gradients[i] = bias_gradients[i] * (1 / ((float) batch_size));
            
            weight_momentum[i] = adam.beta1 * weight_momentum[i] + (1 - adam.beta1) * weight_gradients[i];
            weight_variance[i] = adam.beta2 * weight_variance[i] + (1 - adam.beta2) * matrix<CPU>::square(weight_gradients[i]);
            
            matrix<CPU> weight_momentum_comp = weight_momentum[i] * (1 / (1-std::pow(adam.beta1, ep))); 
            matrix<CPU> weight_variance_comp = weight_variance[i] * (1 / (1-std::pow(adam.beta2, ep))); 

            weight_matrices[i] -= adam.lr * (weight_momentum_comp  % matrix<CPU>::reciprocal(matrix<CPU>::sqrt(weight_variance_comp) + adam.epsilon)); 
            


            bias_momentum[i] = adam.beta1 * bias_momentum[i] + (1 - adam.beta1) * bias_gradients[i];
            bias_variance[i] = adam.beta2 * bias_variance[i] + (1 - adam.beta2) * matrix<CPU>::square(bias_gradients[i]);

            matrix<CPU> bias_momentum_comp = bias_momentum[i] * (1 / (1-std::pow(adam.beta1, ep))); 
            matrix<CPU> bias_variance_comp = bias_variance[i] * (1 / (1-std::pow(adam.beta2, ep))); 

            bias_matrices[i] -= adam.lr * bias_momentum_comp  % matrix<CPU>::reciprocal(matrix<CPU>::sqrt(bias_variance_comp) + adam.epsilon);
        }

    }

}

void neuralnetwork<CPU>::performance(dataset<CPU> &ds, std::string name)
{

    double rmsqe = 0;
    double accuracy = 0;

    
    #pragma omp parallel for reduction(+ : rmsqe, accuracy) num_threads(4)
    for(size_t i = 0; i < ds.input.size(); i++)
    {
        matrix<CPU> pred = run(ds.input[i]);
        matrix<CPU> diff = (pred - ds.expected[i]);
        rmsqe += diff.L2();
        accuracy += (pred.argmax() == ds.expected[i].argmax());
    }   

    rmsqe = std::sqrt(rmsqe / ds.input.size());
    accuracy /= ds.input.size();

    std::cout << "[PERFORMANCE RESULTS FOR DATASET : " << name << "]" << std::endl;
    std::cout << "[ => Accuracy : " << accuracy << "]" <<  std::endl;
    std::cout << "[ => RMSQE : " << rmsqe << "]" << std::endl;

}

void neuralnetwork<CPU>::performance(dataset<CPU> &ds)
{
    performance(ds, "");
}

void neuralnetwork<CPU>::save_weights(const std::string &filename)
{
    std::ofstream file(filename);

    if(!file.is_open())
        throw std::runtime_error("save_weights : File could not be created!");

    

}
