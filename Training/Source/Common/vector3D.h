#ifndef VECTOR3D_
#define VECTOR3D_

#include <iostream>
#include <cmath>
#include <type_traits>
#include <array>
#include <stdexcept>

template <typename T = double>
class Vector3D {
private:
    std::array<T, 3> data;

    // Helper for comparing floating point types
    template <typename U = T>
    typename std::enable_if<std::is_floating_point<U>::value, bool>::type
        almost_equal(U a, U b, double epsilon = 1e-9) const {
        return std::abs(a - b) < epsilon;
    }

    // Helper for comparing non-floating point types
    template <typename U = T>
    typename std::enable_if<!std::is_floating_point<U>::value, bool>::type
        almost_equal(U a, U b, double = 0) const {
        return a == b;
    }

public:
    // Constructors
    Vector3D() : data{ T(0), T(0), T(0) } {}

    Vector3D(T x, T y, T z) : data{ x, y, z } {}

    Vector3D(const Vector3D<T>& other) : data(other.data) {}

    // Constructor from array
    explicit Vector3D(const T arr[3]) : data{ arr[0], arr[1], arr[2] } {}

    // Constructor from std::array
    explicit Vector3D(const std::array<T, 3>& arr) : data(arr) {}

    // Template copy constructor for conversion between types
    template <typename U>
    explicit Vector3D(const Vector3D<U>& other) :
        data{ T(other[0]), T(other[1]), T(other[2]) } {}

    // Destructor
    ~Vector3D() {}

    // Element access
    T& operator[](size_t index) {
        if (index >= 3) throw std::out_of_range("Index out of range");
        return data[index];
    }

    const T& operator[](size_t index) const {
        if (index >= 3) throw std::out_of_range("Index out of range");
        return data[index];
    }

    // Convenience accessors for x, y, z
    T x() const { return data[0]; }
    T y() const { return data[1]; }
    T z() const { return data[2]; }

    void x(T val) { data[0] = val; }
    void y(T val) { data[1] = val; }
    void z(T val) { data[2] = val; }

    // Access to the internal array
    const std::array<T, 3>& getArray() const { return data; }

    // Assignment operator
    Vector3D<T>& operator=(const Vector3D<T>& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }

    // Compound assignment operators
    Vector3D<T>& operator+=(const Vector3D<T>& other) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        data[2] += other.data[2];
        return *this;
    }

    Vector3D<T>& operator-=(const Vector3D<T>& other) {
        data[0] -= other.data[0];
        data[1] -= other.data[1];
        data[2] -= other.data[2];
        return *this;
    }

    Vector3D<T>& operator*=(T scalar) {
        data[0] *= scalar;
        data[1] *= scalar;
        data[2] *= scalar;
        return *this;
    }

    Vector3D<T>& operator/=(T scalar) {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero");
        }
        data[0] /= scalar;
        data[1] /= scalar;
        data[2] /= scalar;
        return *this;
    }

    // Arithmetic operators
    Vector3D<T> operator+(const Vector3D<T>& other) const {
        return Vector3D<T>(
            data[0] + other.data[0],
            data[1] + other.data[1],
            data[2] + other.data[2]
        );
    }

    Vector3D<T> operator-(const Vector3D<T>& other) const {
        return Vector3D<T>(
            data[0] - other.data[0],
            data[1] - other.data[1],
            data[2] - other.data[2]
        );
    }

    Vector3D<T> operator-() const {
        return Vector3D<T>(-data[0], -data[1], -data[2]);
    }

    Vector3D<T> operator*(T scalar) const {
        return Vector3D<T>(data[0] * scalar, data[1] * scalar, data[2] * scalar);
    }

    Vector3D<T> operator/(T scalar) const {
        if (scalar == T(0)) {
            throw std::invalid_argument("Division by zero");
        }
        return Vector3D<T>(data[0] / scalar, data[1] / scalar, data[2] / scalar);
    }

    // Comparison operators
    bool operator==(const Vector3D<T>& other) const {
        return almost_equal(data[0], other.data[0]) &&
            almost_equal(data[1], other.data[1]) &&
            almost_equal(data[2], other.data[2]);
    }

    bool operator!=(const Vector3D<T>& other) const {
        return !(*this == other);
    }

    // Vector operations
    T dot(const Vector3D<T>& other) const {
        return data[0] * other.data[0] +
            data[1] * other.data[1] +
            data[2] * other.data[2];
    }

    Vector3D<T> cross(const Vector3D<T>& other) const {
        return Vector3D<T>(
            data[1] * other.data[2] - data[2] * other.data[1],
            data[2] * other.data[0] - data[0] * other.data[2],
            data[0] * other.data[1] - data[1] * other.data[0]
        );
    }

    T magnitude() const {
        return std::sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
    }

    T lengthSquared() const {
        return data[0] * data[0] + data[1] * data[1] + data[2] * data[2];
    }

    Vector3D<T> normalize() const {
        T mag = magnitude();
        if (almost_equal(mag, T(0))) {
            return Vector3D<T>();
        }
        return Vector3D<T>(data[0] / mag, data[1] / mag, data[2] / mag);
    }

    // Static methods
    static Vector3D<T> cross3D(const Vector3D<T>& a, const Vector3D<T>& b) {
        return Vector3D<T>(
            a.data[1] * b.data[2] - a.data[2] * b.data[1],
            a.data[2] * b.data[0] - a.data[0] * b.data[2],
            a.data[0] * b.data[1] - a.data[1] * b.data[0]
        );
    }


    // Friend functions for iostreams
    friend std::ostream& operator<<(std::ostream& os, const Vector3D<T>& v) {
        os << "(" << v.data[0] << ", " << v.data[1] << ", " << v.data[2] << ")";
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Vector3D<T>& v) {
        is >> v.data[0] >> v.data[1] >> v.data[2];
        return is;
    }

    // Friend function for scalar-vector multiplication (scalar * vector)
    friend Vector3D<T> operator*(T scalar, const Vector3D<T>& v) {
        return v * scalar;
    }
};

// Some common type aliases
using Vector3Df = Vector3D<float>;
using Vector3Dd = Vector3D<double>;
using Vector3Di = Vector3D<int>;

#endif 