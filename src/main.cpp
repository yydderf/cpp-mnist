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
            // z = w * a + b
            this->weights[i - 1].set_size(sizes[i], sizes[i - 1]).randu();
            this->nabla_b[i - 1].set_size(sizes[i], 1);
            this->nabla_w[i - 1].set_size(sizes[i], sizes[i - 1]);
            this->delta_nabla_b[i - 1].set_size(sizes[i], 1);
            this->delta_nabla_w[i - 1].set_size(sizes[i], sizes[i - 1]);
        }
    }

    /**
     * Feedforward the input, then output the prediction of the image
     */
    arma::Mat<double> feedforward(arma::Mat<double> a)
    {
        arma::Mat<double> z;
        for (int i = 0; i < this->num_layers - 1; ++i) {
            z = this->weights[i] * a + this->biases[i];
            a = z.for_each( [](arma::mat::elem_type &val) { return sigmoid(val); } );
        }
        return a;
    }

    /**
     * Stochastic Gradient Descent - the learning algorithm
     */
    void SGD(const std::vector<arma::Mat<uint8_t>> &training_data, uint32_t epochs, uint32_t mini_batch_size, double eta, const std::vector<arma::Mat<uint8_t>> &test_data)
    {
    }

    /**
     * Batching training samples to speed up the training process
     * @param mini_batch A small fraction of the entire training samples
     * @param eta Learning rate
     */
    void update_mini_batch(const std::span<Data> &mini_batch, double eta)
    {
        // calculate nabla_w and nabla_b
        // update self.weights and self.biases
        // size of mini_batch is m
        // w_k -> w'_k = w_k - eta * 1/m * sum(partial nabla C / partial w_k)
        // b_l -> b'_l = b_l - eta * 1/m * sum(partial nabla C / partial b_l)
        // the latter part is derived using backprop
        size_t m = mini_batch.size();
        for (int i = 0; i < this->num_layers - 1; ++i) {
            this->nabla_b[i].zeros();
            this->nabla_w[i].zeros();
        }
        for (Data data : mini_batch) {
            this->backprop(data);
            for (int i = 0; i < this->num_layers - 1; ++i) {
                this->nabla_b[i] += this->delta_nabla_b[i];
                this->nabla_w[i] += this->delta_nabla_w[i];
            }
        }
        for (int i = 0; i < this->num_layers - 1; ++i) {
            this->weights[i] -= (eta / m) * this->nabla_w[i];
            this->biases[i] -= (eta / m) * this->nabla_b[i];
        }
    }

    /**
     * Backpropagate the error of a single training sample
     * 1. input - set corresponding activation layer
     * 2. feedforward - compute z = w * a' + b and a = sigma(z)
     * 3. output error - compute delta for the last layer
     * 4. backpropagate the error - compute the delta for all the other layers
     * 5. output - derive the impact of each weight and bias based on the each error
     * @param data
     */
    void backprop(Data data)
    {
        // this->delta_nabla_b.zeros();
        // this->delta_nabla_w.zeros();
        arma::Mat<double> activation = data.image;
        arma::Mat<double> z;
        std::vector<arma::Mat<double>> activations(this->num_layers);
        activations[0] = activation;

        std::vector<arma::Mat<double>> zs(this->num_layers - 1);

        // 2. feedforward
        for (int i = 0; i < this->num_layers - 1; ++i) {
            z = this->weights[i] * activation + this->biases[i];
            zs[i] = z;
            z.for_each( [](arma::mat::elem_type &val) {return sigmoid(val); } );
            activation = z;
            activations[i + 1] = activation;
        }

        // 3. output error
        arma::Mat<double> delta = this->cost_derivative(activations.end()[-1], data.label) % 
            zs.end()[-1].for_each( [](arma::mat::elem_type &val) { return sigmoid_prime(val); } );
        this->delta_nabla_b.end()[-1] = delta;
        this->delta_nabla_w.end()[-1] = delta * activations.end()[-2].t();

        // 4. & 5.
        for (int i = 2; i < this->num_layers; ++i) {
            delta = (this->weights.end()[-i + 1].t() * delta) %
                zs.end()[-i].for_each( [](arma::mat::elem_type &val) { return sigmoid_prime(val); } );
            // transform should work, but somehow the values after applying sigmoid_prime are not correct
            // delta = (this->weights.end()[-i + 1].t() * delta) %
            //     zs.end()[-i].transform( [](double val) { return sigmoid_prime(val); } );
            this->delta_nabla_b.end()[-i] = delta;
            this->delta_nabla_w.end()[-i] = delta * activations.end()[-i - 1].t();
        }
    }

    void evaluate()
    {
    }

    /**
     * Cost derivative = nabel_a C
     * partial derivative of C with respect to a
     * @param out Output of the neural network
     * @param label Ground truth
     */
    arma::Mat<double> cost_derivative(arma::Mat<double> &out, arma::Mat<double> &label)
    {
        return (out - label);
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
    std::vector<Data> data;
    std::span<Data> training_data, testing_data;
    DataHandler dh(
        data,
        "../res/dataset/train-images.idx3-ubyte",
        "../res/dataset/train-labels.idx1-ubyte"
    );
    dh.split(data, 0.85, training_data, testing_data);
    dh.display_mnist(training_data[0].image);
    dh.display_mnist(testing_data[0].image);

    // set the number of hidden layer neurons to be 30
    std::vector<uint32_t> sizes = {dh.image_size, 30, 10};
    Network network(sizes);
    // network.SGD(training_data, 30, 10, 3.0, test_data)
    
    return 0;
}
