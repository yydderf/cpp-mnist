#include "misc.hpp"

#include <cmath>

double sigmoid(double z)
{
    return 1.0/(1.0 + std::exp(-z));
}

double sigmoid_prime(double z)
{
    return sigmoid(z) * (1 - sigmoid(z));
}
