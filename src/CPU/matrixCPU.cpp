#include "matrixCPU.h"
#include <random>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <algorithm>


matrix<CPU>::matrix()
    : r(0), c(0), n(0)
{}


matrix<CPU>::matrix(const matrix<CPU> &other) 
    : r(other.r), c(other.c), n(other.n), data(other.data)
{}


matrix<CPU>::matrix(matrix<CPU> &&other) noexcept
    : r(other.r), c(other.c), n(other.n), data(std::move(other.data))
{
    other.n = 0;
    other.r = 0;
    other.c = 0;
}

matrix<CPU>::matrix(const size_t rows, const size_t columns) :r(rows), c(columns)
{
    this->n = r * c;
    this->data.resize(n);
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, const std::vector<float> &values) : r(rows), c(columns)
{
    n = r * c;
    if(values.size() != n)
        throw std::runtime_error("Vector size does not match matrix size");

    this->data = values;
}


matrix<CPU>::matrix(const size_t rows, const size_t columns, float val) : matrix<CPU>(rows, columns)
{
    for(int i = 0; i < this->n; i++)
        this->data[i] = val;
}

matrix<CPU>::matrix(const size_t rows, const size_t columns, float start, float end) : matrix<CPU>(rows, columns)
{
    #pragma omp parallel
    {
        std::mt19937 gen(
            std::random_device{}() + omp_get_thread_num()
        );
        std::uniform_real_distribution<float> dist(start, end);

        #pragma omp for
        for (size_t i = 0; i < this->n; ++i)
        {
            this->data[i] = dist(gen);
        }
    }
}

size_t matrix<CPU>::rows() const
{
    return this->r;
}

size_t matrix<CPU>::columns() const
{
    return this->c;
}

size_t matrix<CPU>::size() const
{
    return this->n;
}

std::vector<float> &matrix<CPU>::raw()
{
    return this->data;
}

std::vector<float> matrix<CPU>::raw_copy()
{
    return this->data;
}

double matrix<CPU>::L1()
{
    double res = 0;
    for(size_t i = 0; i < n; i++)
        res += std::abs(data[i]);
    return static_cast<float>(res);
}

double matrix<CPU>::L2()
{
    double res = 0;
    for(size_t i = 0; i < n; i++)
        res += data[i] * data[i];
    return static_cast<float>(std::sqrt(res));
}

size_t matrix<CPU>::argmax()
{
    auto max = max_element(data.begin(), data.end());
    return distance(data.begin(), max); 
}

size_t matrix<CPU>::argmin()
{
    auto max = min_element(data.begin(), data.end());
    return distance(data.begin(), max); 
}

// --------------------------------- OPERATOR ------------------------------

matrix<CPU>& matrix<CPU>::operator=(const matrix<CPU>& other)
{
    if (this != &other) {
        r = other.r;
        c = other.c;
        n = other.n;
        data = other.data;  
    }
    return *this;
}

matrix<CPU>& matrix<CPU>::operator=(matrix<CPU>&& other) noexcept
{
    if (this != &other) {
        r = other.r;
        c = other.c;
        n = other.n;
        data  = std::move(other.data);  
        
        other.r = other.c = other.n = 0;
    }
    return *this;
}

const float &matrix<CPU>::operator[](size_t index) const
{
    return this->data[index];
}

float &matrix<CPU>::operator[](size_t index)
{
    return this->data[index];
}

matrix<CPU> matrix<CPU>::operator%(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, this->c);
    
    matrix<CPU>::hadamard(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, this->c);
    matrix<CPU>::add(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+(const float &a) const
{
    matrix<CPU> result(this->r, this->c);
    for(int i = 0; i < n; i++)
        result[i] = this->data[i] +  a; 
    return result;
}

matrix<CPU> matrix<CPU>::operator-(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, this->c);
    matrix<CPU>::sub(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator-(const float &a) const
{
    matrix<CPU> result(this->r, this->c);
    for(int i = 0; i < n; i++)
        result[i] = this->data[i] - a; 
    return result;
}

matrix<CPU> matrix<CPU>::operator*(const float &a) const
{
    matrix<CPU> result(this->r, this->c);
    matrix<CPU>::scale(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator*(const matrix<CPU> &a) const
{
    matrix<CPU> result(this->r, a.columns()); 
    matrix<CPU>::mat_mul(*this, a, result);
    return result;
}

matrix<CPU> matrix<CPU>::operator+=(const matrix<CPU> &a) 
{
    matrix<CPU>::add(*this, a, *this);
    return *this;
}

matrix<CPU> matrix<CPU>::operator-=(const matrix<CPU> &a) 
{
    matrix<CPU>::sub(*this, a, *this);
    return *this;
}

void matrix<CPU>::mat_mul(const matrix &a, const matrix &b, matrix &result)
{
    // 1. Validierung (bleibt gleich)
    if(a.columns() != b.rows() || result.columns() != b.columns() || result.rows() != a.rows())
        throw std::runtime_error("matmul : matrix shapes do not match.");

    size_t rows = a.rows();
    size_t cols = b.columns();
    size_t inner = a.columns(); // gleich b.rows()

    // 2. Initialisierung: Resultat auf 0 setzen
    result.set(0); 

    // 3. Effiziente 3-Schleifen-Form (Cache-freundlich ohne extra Transpose)
    // Wir tauschen die Schleifenreihenfolge auf row -> inner -> col
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < rows; i++) {
        for(size_t k = 0; k < inner; k++) {
            float temp = a[i * inner + k]; 
            for(size_t j = 0; j < cols; j++) {
                result[i * cols + j] += temp * b[k * cols + j];
            }
        }
    }
}

void matrix<CPU>::mat_mul_transposed(const matrix& a, const matrix &b, matrix& result)
{

    /*
    Transposes b bevore doing matrix multiplication.
     => A * B^T 
    */ 

    //if(a.rows() != b.rows())
        //throw std::runtime_error("mat_mul_transposed : matrix shapes do not match.");

    size_t cols = result.columns();
    size_t rows = result.rows();
    
    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col < cols; col++)
        {
            float sum = 0;
            size_t a_start = row * a.columns();
            size_t b_start = col * a.columns();

            for(int i = 0; i < a.columns(); i++)
                sum += a[a_start + i] * b[b_start + i];

            result[row * cols + col] = sum;
        }
    }
}

void matrix<CPU>::transpose(const matrix& a, matrix& result)
{

    if(a.columns() != result.rows() && a.rows() != result.columns())
        throw std::runtime_error("transpose : matrix shapes do not match.");

    size_t columns = a.columns();
    size_t rows = a.rows();

    for(int row = 0; row < rows; row++)
    {
        for(int col = 0; col< columns; col++)
        {
            result[col * result.columns() + row] = a[row * columns + col];
        }
    }
}

matrix<CPU> matrix<CPU>::transpose()
{
    matrix<CPU> result(this->c, this->r);
    transpose(*this, result);
    return result;
}


void matrix<CPU>::print()
{
    
    for(int row = 0; row < r; row++)
    {
        for(int col = 0; col< c; col++)
        {
            std::cout << this->data[row * c + col] << " ";
        }
        std::cout << std::endl;
    }
}

void matrix<CPU>::print_size()
{
    std::cout << this->r << " " << this->c << std::endl;
}

void matrix<CPU>::set(float val)
{
    for(int i = 0; i < n; i++)
        data[i] = val;
}

void matrix<CPU>::insert_row(size_t row_pos, float val)
{

    if (row_pos > r) {
        throw std::out_of_range("insert_row: row_pos out of range");
    }

    const size_t insert_index = row_pos * c;

    data.insert(
        data.begin() + insert_index,
        c,
        val
    );
    ++r;
    n = r * c;
} 

void matrix<CPU>::remove_row(size_t row_pos)
{
    if (row_pos > r) {
        throw std::out_of_range("remove_row: row_pos out of range");
    }
    const size_t remove_index = row_pos * c;

    data.erase(
        data.begin() + remove_index,
        data.begin() + remove_index + c
    );
    --r;
    n = r * c;
}


void matrix<CPU>::hadamard(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{   
    if( !(a.rows() == b.rows() && b.rows() == result.rows() && a.columns() == b.columns() && b.columns() == result.columns()) )
        throw std::runtime_error("Matrix shapes do not match for the hadamard product. They need to be the same");

    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] * b[i];

}

void matrix<CPU>::add(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( !(a.rows() == b.rows() && b.rows() == result.rows() && a.columns() == b.columns() && b.columns() == result.columns()) )
        throw std::runtime_error("Matrix shapes do not match for the tensor addition. They need to be the same");
    
    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] + b[i];
}

void matrix<CPU>::sub(const matrix<CPU> &a, const matrix<CPU> &b, matrix<CPU> &result)
{
    if( !(a.rows() == b.rows() && b.rows() == result.rows() && a.columns() == b.columns() && b.columns() == result.columns()) )
        throw std::runtime_error("Matrix shapes do not match for the tensor substraction. They need to be the same");

    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] - b[i];
}

void matrix<CPU>::scale(const matrix<CPU> &a, const float value, matrix<CPU> &result)
{
    if( !(a.rows() == result.rows() && a.columns() == result.columns()) )
        throw std::runtime_error("Matrix shapes do not match for the tensor scalar. They need to be the same");
            
    for(int i = 0; i < a.size(); i++)
        result[i] = a[i] * value;

}

matrix<CPU> matrix<CPU>::sqrt(const matrix<CPU> &a)
{
    matrix<CPU> res = a;
    for(size_t i = 0; i < a.size(); i++)
    {
        res[i] = std::sqrt(a[i]);
    }
    return res;
}

matrix<CPU> matrix<CPU>::square(const matrix<CPU> &a)
{
    matrix<CPU> res = a;
    for(size_t i = 0; i < a.size(); i++)
    {
        res[i] = a[i] * a[i];
    }
    return res;
}

matrix<CPU> matrix<CPU>::reciprocal(const matrix<CPU> &a)
{
    matrix<CPU> res = a;
    for(size_t i = 0; i < a.size(); i++)
    {
        res[i] = 1 /  a[i];
    }
    return res;
}

matrix<CPU> operator*(float val, const matrix<CPU> &a)
{
    return a * val;
}

matrix<CPU> operator+(float val, const matrix<CPU> &a)
{
    return a + val;
}

matrix<CPU> operator-(float val, const matrix<CPU> &a)
{
    return (-1) * a + val;
}
