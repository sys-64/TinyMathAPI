#pragma once
#include <iostream>
#include <array>
#include <algorithm>
#include <type_traits>
#include <cmath>

template <typename T, int Rows, int Cols>
class Matrix {
public:
    std::array<std::array<T, Cols>, Rows> data{};

    // Default constructor (initialize all elements to 0)
    Matrix() {
        for (auto& row : data) {
            row.fill(0);
        }
    }

    // Constructor from initializer list
    Matrix(std::initializer_list<std::initializer_list<T>> values) {
        auto rowIt = values.begin();
        for (int i = 0; i < Rows; ++i) {
            auto colIt = rowIt->begin();
            for (int j = 0; j < Cols; ++j) {
                data[i][j] = *colIt++;
            }
            ++rowIt;
        }
    }

    // Element-wise matrix operations
    Matrix operator+(const Matrix& other) const { return apply(other, std::plus<>()); }
    Matrix operator-(const Matrix& other) const { return apply(other, std::minus<>()); }
    Matrix operator*(const Matrix& other) const { return multiply(other); }

    Matrix& operator+=(const Matrix& other) { return apply_self(other, std::plus<>()); }
    Matrix& operator-=(const Matrix& other) { return apply_self(other, std::minus<>()); }
    Matrix& operator*=(const Matrix& other) { return *this = multiply(other); }

    // Scalar operations
    Matrix operator+(const T& scalar) const { return apply_scalar(scalar, std::plus<>()); }
    Matrix operator-(const T& scalar) const { return apply_scalar(scalar, std::minus<>()); }
    Matrix operator*(const T& scalar) const { return apply_scalar(scalar, std::multiplies<>()); }
    Matrix operator/(const T& scalar) const { return apply_scalar(scalar, std::divides<>()); }

    Matrix& operator+=(const T& scalar) { return apply_scalar_self(scalar, std::plus<>()); }
    Matrix& operator-=(const T& scalar) { return apply_scalar_self(scalar, std::minus<>()); }
    Matrix& operator*=(const T& scalar) { return apply_scalar_self(scalar, std::multiplies<>()); }
    Matrix& operator/=(const T& scalar) { return apply_scalar_self(scalar, std::divides<>()); }

    // Matrix utilities
    Matrix transpose() const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[j][i] = data[i][j];
            }
        }
        return result;
    }

    // Vector transformation (multiply matrix by vector)
    template <typename U = T, int N>
    Vector<U, N> transform(const Vector<U, N>& vec) const {
        static_assert(Rows == N, "Matrix row count must match vector size");
        Vector<U, N> result;
        for (int i = 0; i < Rows; ++i) {
            result[i] = 0;
            for (int j = 0; j < Cols; ++j) {
                result[i] += data[i][j] * vec[j];
            }
        }
        return result;
    }

    // Operators
    bool operator==(const Matrix& other) const { return data == other.data; }
    bool operator!=(const Matrix& other) const { return !(*this == other); }
    std::array<T, Cols>& operator[](size_t index) { return data[index]; }
    const std::array<T, Cols>& operator[](size_t index) const { return data[index]; }

    void print() const {
        for (size_t i = 0; i < Rows; ++i) {
            std::cout << "[ ";
            for (size_t j = 0; j < Cols; ++j) {
                std::cout << data[i][j] << (j < Cols - 1 ? ", " : "");
            }
            std::cout << " ]\n";
        }
    }

private:
    template <typename Op>
    Matrix apply(const Matrix& other, Op op) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[i][j] = op(data[i][j], other[i][j]);
            }
        }
        return result;
    }

    template <typename Op>
    Matrix& apply_self(const Matrix& other, Op op) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                data[i][j] = op(data[i][j], other[i][j]);
            }
        }
        return *this;
    }

    template <typename Op>
    Matrix apply_scalar(const T& scalar, Op op) const {
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                result[i][j] = op(data[i][j], scalar);
            }
        }
        return result;
    }

    template <typename Op>
    Matrix& apply_scalar_self(const T& scalar, Op op) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                data[i][j] = op(data[i][j], scalar);
            }
        }
        return *this;
    }

    Matrix multiply(const Matrix& other) const {
        static_assert(Cols == Rows, "Matrix multiplication requires matrix A's columns to match matrix B's rows.");
        Matrix result;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < other.Cols; ++j) {
                result[i][j] = 0;
                for (int k = 0; k < Cols; ++k) {
                    result[i][j] += data[i][k] * other[k][j];
                }
            }
        }
        return result;
    }
};

// Aliases
template <typename T> using mat2x2 = Matrix<T, 2, 2>;
template <typename T> using mat3x3 = Matrix<T, 3, 3>;
template <typename T> using mat4x4 = Matrix<T, 4, 4>;
template <typename T> using Mat2x2 = Matrix<T, 2, 2>;
template <typename T> using Mat3x3 = Matrix<T, 3, 3>;
template <typename T> using Mat4x4 = Matrix<T, 4, 4>;
template <typename T> using mat2X2 = Matrix<T, 2, 2>;
template <typename T> using mat3X3 = Matrix<T, 3, 3>;
template <typename T> using mat4X4 = Matrix<T, 4, 4>;
template <typename T> using Mat2X2 = Matrix<T, 2, 2>;
template <typename T> using Mat3X3 = Matrix<T, 3, 3>;
template <typename T> using Mat4X4 = Matrix<T, 4, 4>;
template <typename T> using matrix2x2 = Matrix<T, 2, 2>;
template <typename T> using matrix3x3 = Matrix<T, 3, 3>;
template <typename T> using matrix4x4 = Matrix<T, 4, 4>;
template <typename T> using Matrix2x2 = Matrix<T, 2, 2>;
template <typename T> using Matrix3x3 = Matrix<T, 3, 3>;
template <typename T> using Matrix4x4 = Matrix<T, 4, 4>;
template <typename T> using matrix2X2 = Matrix<T, 2, 2>;
template <typename T> using matrix3X3 = Matrix<T, 3, 3>;
template <typename T> using matrix4X4 = Matrix<T, 4, 4>;
template <typename T> using Matrix2X2 = Matrix<T, 2, 2>;
template <typename T> using Matrix3X3 = Matrix<T, 3, 3>;
template <typename T> using Matrix4X4 = Matrix<T, 4, 4>;
