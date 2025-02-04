#include <vector>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>
#include <algorithm>
#include <random>

#include "misc.hpp"

// reference: https://stackoverflow.com/questions/12993941/how-can-i-read-the-mnist-dataset-with-c

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

inline uint32_t swap_endian(uint32_t i)
{
    // reference: https://en.wikipedia.org/wiki/Endianness
    // 0xDDCCBBAA
    // 0xAABBCCDD
    return (((i >> 24) & 0xFF) |
            (((i >> 16) & 0xFF) << 8) |
            (((i >> 8) & 0xFF) << 16) |
            ((i & 0xFF) << 24));
}

// DataHandler::DataHandler(std::vector<arma::Mat<uint8_t>> &images, std::vector<arma::Mat<uint8_t>> &labels, const char *image_filename, const char *label_filename)
DataHandler::DataHandler(std::vector<Data> &data, const char *image_filename, const char *label_filename)
{
    std::ifstream image_fd(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_fd(label_filename, std::ios::in | std::ios::binary);

    uint32_t magic;

    image_fd.read(reinterpret_cast<char *>(&magic), 4);
    if (swap_endian(magic) != 0x00000803) {
        std::cerr << "[!] Image file magic number error" << std::endl;
        std::exit(1);
    }

    label_fd.read(reinterpret_cast<char *>(&magic), 4);
    if (swap_endian(magic) != 0x00000801) {
        std::cerr << "[!] Label file magic number error" << std::endl;
        std::exit(1);
    }

    image_fd.read(reinterpret_cast<char *>(&(this->num_images)), 4);
    this->num_images = swap_endian(this->num_images);

    label_fd.read(reinterpret_cast<char *>(&(this->num_labels)), 4);
    this->num_labels = swap_endian(this->num_labels);

    if (this->num_images != this->num_labels) {
        std::cerr << "[!] Number of images and labels are not matched." << std::endl;
        std::cerr << "\tImage: " << this->num_images << ", Labels: " << this->num_labels << std::endl;
    }

    image_fd.read(reinterpret_cast<char *>(&(this->rows)), 4);
    image_fd.read(reinterpret_cast<char *>(&(this->cols)), 4);

    this->rows = swap_endian(this->rows);
    this->cols = swap_endian(this->cols);

    image_size = this->rows * this->cols;

    data.resize(this->num_images);

    uint8_t *tmp_image = new uint8_t[this->image_size];
    // arma::Mat<uint8_t> tmp_image(this->image_size, 1);
    uint8_t tmp_label;

    for (int i = 0; i < this->num_images; ++i) {
        data[i].image.set_size(this->image_size, 1);
        data[i].label.zeros(10, 1);

        // image_fd.read(reinterpret_cast<char *>(data[i].image.memptr()), this->image_size);
        image_fd.read(reinterpret_cast<char *>(tmp_image), this->image_size);
        label_fd.read(reinterpret_cast<char *>(&(tmp_label)), 1);

        for (int j = 0; j < this->image_size; ++j) {
            data[i].image(j, 0) = static_cast<double>(tmp_image[j]) / 255;
            // data[i].image(j, 0) = static_cast<double>(tmp_image[j]);
        }
        data[i].label(tmp_label, 0) = 1.0;
    }

    delete[] tmp_image;
    image_fd.close();
    label_fd.close();
}

/**
 * Display the image to the terminal
 * @param image The matrix storing the data of an image
 */
void DataHandler::display_mnist(const arma::Mat<double> &image)
{
    for (int i = 0; i < this->image_size; ++i) {
        if (i % this->rows == 0) {
            std::cout << std::endl;
        }
        std::cout << ((image[i] > 127) ? '#' : ' ');
    }
    std::cout << std::endl;
}

/**
 * Split the dataset into training and testing based on the given ratio
 * @param data The entire dataset
 * @param ratio The ratio for training data
 * @param training_data A lightweight reference the training data
 * @param testing_data A lightweight reference the testing data
 */
void DataHandler::split(std::vector<Data> &data, double ratio, std::span<Data> &training_data, std::span<Data> &testing_data)
{
    uint32_t training_size = std::lround(this->num_images * ratio);
    uint32_t testing_size = this->num_images - training_size;

    std::random_device rd;
    std::mt19937 generator(rd());
    std::shuffle(data.begin(), data.end(), generator);
    std::span<Data> s(data);
    training_data = s.subspan(0, training_size);
    testing_data = s.subspan(training_size);
}
