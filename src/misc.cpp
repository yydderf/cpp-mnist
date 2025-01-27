#include <vector>
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

void load_mnist(std::vector<arma::Mat<uint8_t>> &images, std::vector<uint8_t> &labels, const char *image_filename, const char *label_filename)
{
    std::ifstream image_fd(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_fd(label_filename, std::ios::in | std::ios::binary);

    uint32_t magic;
    uint32_t num_images;
    uint32_t num_labels;
    uint32_t rows, cols;
    uint32_t image_size;

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

    image_fd.read(reinterpret_cast<char *>(&num_images), 4);
    num_images = swap_endian(num_images);

    label_fd.read(reinterpret_cast<char *>(&num_labels), 4);
    num_labels = swap_endian(num_labels);

    if (num_images != num_labels) {
        std::cerr << "[!] Number of images and labels are not matched." << std::endl;
    }

    image_fd.read(reinterpret_cast<char *>(&rows), 4);
    image_fd.read(reinterpret_cast<char *>(&cols), 4);

    rows = swap_endian(rows);
    cols = swap_endian(cols);

    image_size = rows * cols;

    images.resize(num_images);
    labels.resize(num_labels);

    for (int i = 0; i < num_images; ++i) {
        images[i].set_size(image_size, 1);
        image_fd.read(reinterpret_cast<char *>(images[i].memptr()), image_size);
    }

    for (int i = 0; i < num_labels; ++i) {
        label_fd.read(reinterpret_cast<char *>(&(labels[i])), 1);
    }
}
