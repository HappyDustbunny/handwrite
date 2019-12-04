import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from read_write_and_plot import read_label_data, read_picture_data, plot_stuff


def main():
    net = initialize_network()
    train_network(net)
    # evaluate_network(net)


class Network:
    def __init__(self, hyper_parameters):
        self.number_of_layers = len(hyper_parameters.nodes_pr_layer)
        self.learning_rate = hyper_parameters.learning_rate
        self.nodes_pr_layer = hyper_parameters.nodes_pr_layer
        self.epochs = hyper_parameters.epochs
        self.layers = []
        self.target = np.zeros(self.nodes_pr_layer[-1])
        self.c = np.zeros(self.nodes_pr_layer[-1])
        self.cost = 0
        self.cost_history = None

        # Initialize and store layers
        for n, number_of_nodes in enumerate(self.nodes_pr_layer):
            layer = Layer(n, number_of_nodes)
            self.layers.append(layer)

        # Initialize weigths
        for n in range(self.number_of_layers):
            # for n, layer in enumerate(self.layers):
            if n > 0:
                self.layers[n].weights = np.random.rand(self.layers[n - 1].non
                                                        + 1, self.layers[n].non)
                # + 1 because of bias
                # print(n, self.layers[n].weights.shape)

    def load_single_pic(self, single_label, single_picture):
        self.target[single_label] = 1
        self.layers[0].a = single_picture

    def find_A(self, n):
        longer_a = np.append(self.layers[n].a, 1)
        longer_a.shape = (1, self.layers[n].non + 1)
        self.layers[n].A = longer_a
        # self.layers[n].A = np.transpose(longer_a)
        # print(self.layers[n].A.shape)

    def find_c(self):
        # c = (a - T)
        self.c = self.layers[-1].a - self.target

    def find_cost(self):
        # C = ½(a - T)**2
        self.cost = sum(0.5 * (self.layers[-1].a - self.target) ** 2)

    def find_sigma(self, n, value):
        self.layers[n].a = 1 / (1 - np.exp(-value))

    def find_D(self, n):
        # D = a(1 - a)
        self.layers[n].D = self.layers[n].a * (1 - self.layers[n].a)

    # def find_w(self, n):
    #     # Removing the biases from the weights matrices
    #     if n > 0:
    #         self.layers[n].w = self.layers[n].weights[0: -1, : ]
    #         # [0: -1, : ] All rows except the last, all collums

    def find_delta(self, n):
        if n == self.number_of_layers - 1:
            self.find_c()
            self.delta = self.layers[n].D * self.c
            # self.delta = np.matmul(self.layers[n].D, self.c)
            # delta(n-1) = D(n) @ c(n)    n being number of last layer
        else:
            self.layers[n].delta = np.matmul(np.matmul(self.layers[n - 1].D,
                                             self.layers[n].weights[0: -1, :]),
                                             self.layers[n].delta)
            # delta(i) = D(i) @ w(i+1) @ delta(i+1)
            #
            # [0: -1, : ] All rows except the last,
            # all columns
        self.layers[n].delta.shape = (self.layers[n].non, 1)


class Layer(Network):
    def __init__(self, layer_number, number_of_nodes):
        # super().__init__(hyper_parameters)
        self.a = np.zeros(number_of_nodes)
        self.A = np.append(self.a, 1)
        self.weights = None  # With biases
        # self.w = None  # Without biases
        self.D = np.zeros(number_of_nodes)
        self.delta = np.zeros(number_of_nodes)
        # self.sigma = np.zeros(number_of_nodes)
        self.n = layer_number
        self.non = number_of_nodes
        self.dCdW = None


def initialize_network():
    hyper_param = namedtuple('Hyper_Parameters', ['learning_rate',
                                                  'nodes_pr_layer',
                                                  'epochs'], )
    hyper_parameters = hyper_param(0.5,
                                   (784, 16, 16, 10),  # (In, Hidden, ..., Out)
                                   1, )

    net = Network(hyper_parameters)

    return net


def train_network(net):
    # Load training data
    # label_data = read_label_data('train-labels.idx1-ubyte')
    # picture_data = read_picture_data('train-images.idx3-ubyte')
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')

    # Start training
    n = 0
    number_of_iterations = 500
    net.cost_history = np.zeros((number_of_iterations + 1,))
    for single_label, single_picture in zip(label_data, picture_data):
        net.load_single_pic(int(single_label), single_picture)
        forward_pass(net)
        net.cost_history[n] = net.cost
        back_prop(net)
        update_weights(net)
        if not n % 10:
            plot_stuff(int(single_label), single_picture, net.layers[-1].a)
        n += 1
        if n > number_of_iterations:
            break

    plt.plot(net.cost_history)
    plt.show()

    store_weights(net)


def forward_pass(net):
    for n, layer in enumerate(net.layers):
        if n == 0:
            net.find_A(n)
        # elif n < net.layers[n].nol:
        elif n < net.number_of_layers:
            A_matmul_w = np.matmul(net.layers[n - 1].A,
                                   net.layers[n].weights)
            # print(A_matmul_w, A_matmul_w.shape)
            net.find_sigma(n, A_matmul_w)
            net.find_A(n)
            net.find_D(n)
            # net. find_w(n)  # Maybe just call weights[0: -1, : ] when needed?
            net.find_delta(n)
        else:
            net.find_cost()


def back_prop(net):
    for n, layer in enumerate(net.layers):
        N = net.number_of_layers - 1  # N - n goes trough layers backwards, but
        # the number of layers need to be adjusted
        # for counting from 0
        if N - n > 0:
            # print('delta', net.layers[N - n].delta.shape, '\nA', net.layers[N - n - 1].A)
            net.layers[N - n].dCdW = np.matmul(net.layers[N - n].delta,
                                               net.layers[N - n - 1].A)
            # print(net.layers[N - n].dCdW, net.layers[N - n].dCdW.shape)


def update_weights(net):
    for n, layer in enumerate(net.layers):
        if n > 0:
            net.layers[n].weights -= np.transpose(net.learning_rate
                                                  * net.layers[n].dCdW)


def store_weights(net):
    pass


def load_weights():
    pass


def evaluate_network(net):
    pass
    # Load evaluation data
    # label_data = read_label_data('t10k-labels.idx1-ubyte')
    # picture_data = read_picture_data('t10k-images.idx3-ubyte')
    # load_weights()


if __name__ == '__main__':
    main()
