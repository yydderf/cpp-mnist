#pragma once

#include <cmath>
#include <vector>
#include <armadillo>

double sigmoid(double z);
double sigmoid_prime(double z);

class DataHandler {
public:
    DataHandler(std::vector<arma::Mat<uint8_t>> &images, std::vector<uint8_t> &labels, const char *image_filename, const char *label_filename);
    void display_mnist(const arma::Mat<uint8_t> &image);
    uint32_t num_images;
    uint32_t num_labels;
    uint32_t rows, cols;
    uint32_t image_size;
};

