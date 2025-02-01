#pragma once

#include <cmath>
#include <vector>
#include <armadillo>
#include <span>

double sigmoid(double z);
double sigmoid_prime(double z);

struct Data {
    arma::Mat<double> image;
    arma::Mat<double> label;
};

class DataHandler {
public:
    DataHandler(std::vector<Data> &data, const char *image_filename, const char *label_filename);
    void display_mnist(const arma::Mat<double> &image);
    void split(std::vector<Data> &data, double ratio, std::span<Data> &training_data, std::span<Data> &testing_data);
    uint32_t num_images;
    uint32_t num_labels;
    uint32_t rows, cols;
    uint32_t image_size;
};

