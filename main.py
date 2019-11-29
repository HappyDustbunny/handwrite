import numpy as np
from collections import namedtuple
from read_write_and_plot import read_label_data, read_picture_data, plot_stuff

def main():

    net = initialize_network()
    train_network(net)
    #evaluate_network(net)


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
        self.dCdW = None

        # Initialize and store layers
        for n, number_of_nodes in enumerate(self.nodes_pr_layer):
            layer = Layer(n, number_of_nodes, self.number_of_layers)
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
        self.layers[n].A = np.transpose(longer_a)
        print(self.layers[n].A.shape)


    def find_c(self):
        self.c = self.layers[-1].a - self.target  # c = (a - T)


    def find_cost(self):
        self.cost = 0.5*(self.layers[-1].a - self.target)**2  # C = ½(a - T)**2


    def find_D(self, layer_number):
        self.layers[layer_number].D = self.a*(1 - self.a)  # D = a(1 - a)


    def find_delta(self, layer_number):
        if self.n == self.number_of_layers:
            find_c()
            self.delta = np.matmul(self.layers[layer_number].D, self.c)
            # delta(n-1) = D(n) @ c(n)    n being number of last layer
        else:
            self.delta = np.matmul(np.matmul(self.layers[layer_number - 1].D,
                                   self.layers[layer_number].weights),
                                   self.layers[layer_number].delta)
                                   # delta(i) = D(i) @ w(i+1) @ delta(i+1)


    def find_sigma(self, n, value):
        self.layers[n].a = 1/(1 - np.exp(-value))
        self.find_A(n)


class Layer(Network):
    def __init__(self, layer_number, number_of_nodes, number_of_layers):
        # super().__init__(hyper_parameters)
        self.a = np.zeros(number_of_nodes)
        self.A = np.append(self.a, 1)
        self.weights = None
        self.D = np.zeros(number_of_nodes)
        self.delta = np.zeros(number_of_nodes)
        #self.sigma = np.zeros(number_of_nodes)
        self.n = layer_number
        self.non = number_of_nodes
        self.nol = number_of_layers


def initialize_network():
        hyper_param = namedtuple('Hyper_Parameters', ['learning_rate',
                                                      'nodes_pr_layer',
                                                      'epochs'],)
        hyper_parameters = hyper_param(0.5,
                                       (784, 16, 16, 10),  # (In, Hidden, ..., Out)
                                       1,)

        net = Network(hyper_parameters)

        return net


def train_network(net):
    # Load training data
    # label_data = read_label_data('train-labels.idx1-ubyte')
    # picture_data = read_picture_data('train-images.idx3-ubyte')
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')

    n = 0
    for single_label, single_picture in zip(label_data, picture_data):
        net.load_single_pic(int(single_label), single_picture)
        plot_stuff(int(single_label), single_picture)
        forward_pass(net)
        back_prop(net)
        update_weights(net)
        n += 1
        if n > 5:
            break

    store_weights()


def forward_pass(net):
    for n, layer in enumerate(net.layers):
        if n == 0:
            net.find_A(n)
        elif n < net.layers[n].nol:
            A_matmul_w = np.matmul(net.layers[n-1].A, net.layers[n].weights)
            print(A_matmul_w, A_matmul_w.shape)
            net.find_sigma(n, A_matmul_w)
        else:
            net.find_cost()


def back_prop(net):
    pass


def update_weights(net):
    pass


def store_weights(net):
    pass


def evaluate_network(net):
    # Load evaluation data
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')
    load_weights()


if __name__ == '__main__':
    main()
