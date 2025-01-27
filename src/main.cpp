#include <iostream>
#include <vector>
#include <armadillo>

#include "misc.hpp"

class Network {
public:
    Network(std::vector<int> sizes)
    {
        this->num_layers = sizes.size();
        this->sizes = sizes;

        this->biases.resize(num_layers - 1);
        this->weights.resize(num_layers - 1);

        for (int i = 1; i < num_layers; ++i) {
            this->biases[i - 1].set_size(sizes[i], 1).randu();
            this->weights[i - 1].set_size(sizes[i - 1], sizes[i]).randu();
        }
    }

    arma::Mat<double> feedforward(arma::Mat<double> a)
    {
        return a;
    }

    void SGD()
    {
    }

    void update_mini_batch(double eta)
    {
    }

    void backprop()
    {
    }

    void evaluate()
    {
    }

    void cost_derivative()
    {
    }

private:
    int num_layers;
    std::vector<int> sizes;
    std::vector<arma::Mat<double>> biases;
    std::vector<arma::Mat<double>> weights;
};

int main()
{
    // std::vector<int> sizes = {2, 3, 1};
    // Network network(sizes);
    // arma::mat A(4, 5, arma::fill::randu);
    // arma::mat B(4, 5, arma::fill::randu);

    // std::cout << A * B.t() << std::endl;
    std::vector<arma::Mat<uint8_t>> images;
    std::vector<uint8_t> labels;
    load_mnist(
        images,
        labels,
        "../res/dataset/t10k-images.idx3-ubyte",
        "../res/dataset/t10k-labels.idx1-ubyte"
    );
    return 0;
}
