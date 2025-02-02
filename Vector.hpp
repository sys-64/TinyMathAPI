#pragma once
#include <iostream>
#include <array>
#include <algorithm>
#include <type_traits>

template <typename T, int N>
class Vector {
public:
    std::array<T, N> data{};

    Vector() { data.fill(0); }
    Vector(std::initializer_list<T> values) {
        std::copy_n(values.begin(), std::min(N, static_cast<int>(values.size())), data.begin());
    }

    // Element-wise vector operations
    Vector operator+(const Vector& other) const { return apply(other, std::plus<>()); }
    Vector operator-(const Vector& other) const { return apply(other, std::minus<>()); }
    Vector operator*(const Vector& other) const { return apply(other, std::multiplies<>()); }
    Vector operator/(const Vector& other) const { return apply(other, std::divides<>()); }

    Vector& operator+=(const Vector& other) { return apply_self(other, std::plus<>()); }
    Vector& operator-=(const Vector& other) { return apply_self(other, std::minus<>()); }
    Vector& operator*=(const Vector& other) { return apply_self(other, std::multiplies<>()); }
    Vector& operator/=(const Vector& other) { return apply_self(other, std::divides<>()); }

    // Scalar operations
    Vector operator+(const T& scalar) const { return apply_scalar(scalar, std::plus<>()); }
    Vector operator-(const T& scalar) const { return apply_scalar(scalar, std::minus<>()); }
    Vector operator*(const T& scalar) const { return apply_scalar(scalar, std::multiplies<>()); }
    Vector operator/(const T& scalar) const { return apply_scalar(scalar, std::divides<>()); }

    Vector& operator+=(const T& scalar) { return apply_scalar_self(scalar, std::plus<>()); }
    Vector& operator-=(const T& scalar) { return apply_scalar_self(scalar, std::minus<>()); }
    Vector& operator*=(const T& scalar) { return apply_scalar_self(scalar, std::multiplies<>()); }
    Vector& operator/=(const T& scalar) { return apply_scalar_self(scalar, std::divides<>()); }

    // Vector utilities
    T dot(const Vector& other) const {
        T result = 0;
        for (size_t i = 0; i < N; i++)
            result += data[i] * other.data[i];
        return result;
    }

    T magnitude() const {
        return std::sqrt(dot(*this));
    }

    Vector normalized() const {
        T mag = magnitude();
        return (mag > 0) ? *this / mag : *this;
    }

    static T distance(const Vector& a, const Vector& b) {
        return (a - b).magnitude();
    }

    Vector& normalize() {
        *this = normalized();
        return *this;
    }

    // Cross Product (only for Vec3)
    template <typename U = T>
    Vector cross(const Vector<U, 3>& other) const {
        static_assert(N == 3, "Cross product is only valid for 3D vectors.");
        return Vector{
            data[1] * other.data[2] - data[2] * other.data[1],
            data[2] * other.data[0] - data[0] * other.data[2],
            data[0] * other.data[1] - data[1] * other.data[0]
        };
    }

    // Clamp values within min-max range
    Vector clamp(const T& minVal, const T& maxVal) const {
        Vector result;
        for (size_t i = 0; i < N; i++)
            result.data[i] = std::clamp(data[i], minVal, maxVal);
        return result;
    }

    // Linear interpolation
    static Vector lerp(const Vector& start, const Vector& end, T t) {
        return start + (end - start) * t;
    }

    // Reflection over a normal
    Vector reflect(const Vector& normal) const {
        return *this - normal * (2 * dot(normal));
    }

    // Operators
    bool operator==(const Vector& other) const { return data == other.data; }
    bool operator!=(const Vector& other) const { return !(*this == other); }
    Vector operator-() const { return *this * -1; }
    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }

    void print() const {
        std::cout << "(";
        for (size_t i = 0; i < N; i++)
            std::cout << data[i] << (i < N - 1 ? ", " : "");
        std::cout << ")\n";
    }

private:
    template <typename Op>
    Vector apply(const Vector& other, Op op) const {
        Vector result;
        std::transform(data.begin(), data.end(), other.data.begin(), result.data.begin(), op);
        return result;
    }

    template <typename Op>
    Vector& apply_self(const Vector& other, Op op) {
        std::transform(data.begin(), data.end(), other.data.begin(), data.begin(), op);
        return *this;
    }

    template <typename Op>
    Vector apply_scalar(const T& scalar, Op op) const {
        Vector result;
        std::transform(data.begin(), data.end(), result.data.begin(), [&](T x) { return op(x, scalar); });
        return result;
    }

    template <typename Op>
    Vector& apply_scalar_self(const T& scalar, Op op) {
        std::transform(data.begin(), data.end(), data.begin(), [&](T x) { return op(x, scalar); });
        return *this;
    }
};

// Aliases
template <typename T> using vec2 = Vector<T, 2>;
template <typename T> using vec3 = Vector<T, 3>;
template <typename T> using vec4 = Vector<T, 4>;
template <typename T> using Vec2 = Vector<T, 2>;
template <typename T> using Vec3 = Vector<T, 3>;
template <typename T> using Vec4 = Vector<T, 4>;
template <typename T> using Vector2 = Vector<T, 2>;
template <typename T> using Vector3 = Vector<T, 3>;
template <typename T> using Vector4 = Vector<T, 4>;
template <typename T> using vector2 = Vector<T, 2>;
template <typename T> using vector3 = Vector<T, 3>;
template <typename T> using vector4 = Vector<T, 4>;