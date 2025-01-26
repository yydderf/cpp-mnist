#pragma once

#include <cmath>

double sigmoid(double z)
{
    // double z = std::static_cast<double>(_z);
    return 1.0 / (1.0 + std::exp(-z));
}

double sigmoid_prime(double z)
{
    // double z = std::static_cast<double>(_z);
    return sigmoid(z) * (1 - sigmoid(z));
}
