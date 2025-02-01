#include <iostream>
#include <vector>
#include <armadillo>

#include "misc.hpp"

class Network {
public:
Network(std::vector<uint32_t> sizes)
    {
        this->num_layers = sizes.size();
        this->sizes = sizes;

        this->biases.resize(num_layers - 1);
        this->weights.resize(num_layers - 1);
        this->nabla_b.resize(num_layers - 1);
        this->nabla_w.resize(num_layers - 1);
        this->delta_nabla_b.resize(num_layers - 1);
        this->delta_nabla_w.resize(num_layers - 1);

        for (int i = 1; i < num_layers; ++i) {
            this->biases[i - 1].set_size(sizes[i], 1).randu();
            this->weights[i - 1].set_size(sizes[i - 1], sizes[i]).randu();
        }
    }

    arma::Mat<double> feedforward(arma::Mat<double> a)
    {
        return a;
    }

    void SGD(const std::vector<arma::Mat<uint8_t>> &training_data, uint32_t epochs, uint32_t mini_batch_size, double eta, const std::vector<arma::Mat<uint8_t>> &test_data)
    {
    }

    void update_mini_batch(const std::vector<Data> &mini_batch, double eta)
    {
        // calculate nabla_w and nabla_b
        // update self.weights and self.biases
        // size of mini_batch is m
        // w_k -> w'_k = w_k - eta * 1/m * sum(partial nabla C / partial w_k)
        // b_l -> b'_l = b_l - eta * 1/m * sum(partial nabla C / partial b_l)
        // the latter part is derived using backprop
        int n = mini_batch.size();
        // this->nabla_b.zeros();
        // this->nabla_w.zeros();
        // for (Data data : mini_batch) {
        // }
        // this->weights;
        // this->biases;
    }

    void backprop(Data data)
    {
        // this->delta_nabla_b.zeros();
        // this->delta_nabla_w.zeros();
        arma::Mat<double> activation = data.image;
        std::vector<arma::Mat<double>> activations(this->num_layers);
        activations[0] = activation;

        std::vector<arma::Mat<double>> zs(this->num_layers - 1);

        double z;

        // for (int i = 0; i < this->num_layers - 1; ++i) {
        //     double z = arma::dot(this->weights[i], activation) + this->biases[i];
        // }
        
    }

    void evaluate()
    {
    }

    void cost_derivative()
    {
    }

private:
    int num_layers;
    std::vector<uint32_t> sizes;
    std::vector<arma::Mat<double>> biases;
    std::vector<arma::Mat<double>> weights;
    std::vector<arma::Mat<double>> nabla_b;
    std::vector<arma::Mat<double>> nabla_w;
    std::vector<arma::Mat<double>> delta_nabla_b;
    std::vector<arma::Mat<double>> delta_nabla_w;
};

int main()
{
    // arma::mat A(4, 5, arma::fill::randu);
    // arma::mat B(4, 5, arma::fill::randu);

    // std::cout << A * B.t() << std::endl;
    std::vector<Data> data;
    std::span<Data> training_data, testing_data;
    DataHandler dh(
        data,
        "../res/dataset/train-images.idx3-ubyte",
        "../res/dataset/train-labels.idx1-ubyte"
    );
    // dh.display_mnist(data[0].image);
    dh.split(data, 0.85, training_data, testing_data);
    // std::cout << training_data.size() << ' ' << testing_data.size() << std::endl;
    dh.display_mnist(training_data[0].image);
    dh.display_mnist(testing_data[0].image);

    // set the number of hidden layer neurons to be 30
    // std::vector<uint32_t> sizes = {dh.image_size, 30, 10};
    // Network network(sizes);
    // network.SGD(training_data, 30, 10, 3.0, test_data)
    
    return 0;
}
