#pragma once


__global__ void matrix_kernel_sum(const float* data, float *result, const size_t mat_size);

__global__ void matrix_kernel_argmax(const float* data, size_t* result, const size_t mat_size);

__global__ void matrix_kernel_argmin(const float* data, size_t* result, const size_t mat_size);

__global__ void matrix_kernel_max(const float* data, float* result, const size_t mat_size);

__global__ void matrix_kernel_min(const float* data, float* result, const size_t mat_size);


__global__ void matrix_kernel_set(float* data, float val, size_t n);

__global__ void matrix_kernel_sqrt(const float* data, float* result, size_t n);

__global__ void matrix_kernel_square(const float* data, float* result, size_t n);

__global__ void matrix_kernel_reciprocal(const float* data, float* result, size_t n);

__global__ void matrix_kernel_exp(const float* data, float* result, size_t n);

__global__ void matrix_kernel_log2(const float* data, float* result, size_t n);

__global__ void matrix_kernel_hadamard(const float *A, const float *B, float *result, const size_t n);

__global__ void matrix_kernel_add(const float *A, const float *B, float *result, const size_t n);

__global__ void matrix_kernel_sub(const float *A, const float *B, float *result, const size_t n);

__global__ void matrix_kernel_scale(const float *A, float *result, const float value, const size_t n);

__global__ void matrix_kernel_add_value(const float *A, float *result, const float value, const size_t n);

__global__ void matrix_kernel_mat_mul(const float *A, const float *B, float *result, const size_t result_rows, const size_t result_cols, const size_t length);


__global__ void matrix_kernel_reduce_sum(const float *A, float *result, const size_t mat_size, const size_t n);

__global__ void matrix_kernel_bcast_add_to_stacked_matrix(const float* A, const float* B, float *result, const size_t mat_size, const size_t n);


__global__ void matrix_kernel_bcast_reversed_mat_mul_to_stacked_matrix(const float *A, const float *B, float *result, const size_t result_rows, const size_t result_cols, const size_t length);

__global__ void matrix_kernel_bcast_mat_mul_to_stacked_matrix(const float *A, const float *B, float *result, const size_t result_rows, const size_t result_cols, const size_t length);

__global__ void matrix_kernel_bcast_scale_to_stacked_matrix(const float* A, const float* B, float *result, const size_t mat_size, const size_t n);

__global__ void matrix_kernel_bcast_hadamard_to_stacked_matrix(const float* A, const float* B, float *result, const size_t mat_size, const size_t n);

__global__ void matrix_kernel_transpose(const float* A, float* result, const size_t result_rows, const size_t result_columns, const size_t n);






__global__ void activation_function_kernel_relu(const float* A, float* result, const size_t n);

__global__ void activation_function_kernel_elu(const float* A, float* result, const float alpha, const size_t n);

__global__ void activation_function_kernel_sigmoid(const float* A, float* result, const size_t n);

__global__ void activation_function_kernel_log_sigmoid(const float* A, float* result, const size_t n);

__global__ void activation_function_kernel_hard_sigmoid(const float* A, float* result, const size_t n);

__global__ void activation_function_kernel_tanh(const float* A, float* result, const size_t n);


__global__ void activation_function_kernel_drelu(const float* A, float* result, const size_t n);

__global__ void activation_function_kernel_delu(const float* A, float* result, const float alpha, const size_t n);

__global__ void activation_function_kernel_dhard_sigmoid(const float* A, float* result, const size_t n);

__global__ void activation_function_kernel_dtanh(const float* A, float* result, const size_t n);





