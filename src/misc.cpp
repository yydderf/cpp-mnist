#include <vector>
#include <iostream>
#include <fstream>
#include <armadillo>

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

DataHandler::DataHandler(std::vector<arma::Mat<uint8_t>> &images, std::vector<uint8_t> &labels, const char *image_filename, const char *label_filename)
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
    }

    image_fd.read(reinterpret_cast<char *>(&(this->rows)), 4);
    image_fd.read(reinterpret_cast<char *>(&(this->cols)), 4);

    this->rows = swap_endian(this->rows);
    this->cols = swap_endian(this->cols);

    image_size = this->rows * this->cols;

    images.resize(this->num_images);
    labels.resize(this->num_labels);

    for (int i = 0; i < this->num_images; ++i) {
        images[i].set_size(this->image_size, 1);
        image_fd.read(reinterpret_cast<char *>(images[i].memptr()), this->image_size);
    }

    for (int i = 0; i < this->num_labels; ++i) {
        label_fd.read(reinterpret_cast<char *>(&(labels[i])), 1);
    }

    image_fd.close();
    label_fd.close();
}

void DataHandler::display_mnist(const arma::Mat<uint8_t> &image)
{
    for (int i = 0; i < this->image_size; ++i) {
        if (i % this->rows == 0) {
            std::cout << std::endl;
        }
        std::cout << ((image[i] > 127) ? '#' : ' ');
    }
    std::cout << std::endl;
}
