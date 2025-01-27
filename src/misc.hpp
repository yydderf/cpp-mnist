#pragma once

#include <cmath>
#include <vector>
#include <armadillo>

double sigmoid(double z);
double sigmoid_prime(double z);
void load_mnist(std::vector<arma::Mat<uint8_t>> &images, std::vector<uint8_t> &labels, const char *image_filename, const char *label_filename);
