import numpy as np
from collections import namedtuple
from read_write_and_plot import read_label_data, read_picture_data, plot_stuff

def main():

    net = initialize_network()
    #train_network(net)
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

        for n, number_of_nodes in enumerate(self.nodes_pr_layer):
            layer = Layer(n, number_of_nodes, self.number_of_layers)
            self.layers.append(layer)


    def load_data(self, target_number, picture):
        self.target[target_number] = 1
        self.layers[0].a = picture

    def initialize_weigths(self):
        for n, layer in enumerate(layers):
            if n > 0:
                self.layers[n].weights = np.random.rand(self.layers[n].non,
                                                        self.layers[n - 1].non)


    def find_c(self):
        self.c = self.layers[-1].a - self.target


    def find_cost(self):
        self.cost = 0.5*(self.layers[-1].a - self.target)**2


    def find_D(self, layer_number):
        self.layers[layer_number].D = self.a*(1 - self.a)


    def find_delta(self, layer_number):
        if self.layer_number == number_of_layers:
            find_c()
            self.sigma = np.matmul(self.layers[layer_number].D, self.c)
        else:
            self.sigma = np.matmul(np.matmul(self.layers[layer_number - 1].D,
                                   self.layers[layer_number].weights),
                                   self.layers[layer_number].delta)


    def find_sigma(self, layer_num):
        self.layers[layer_num].sigma = 1/(1 - exp(-self.layers[layer_num].a))


class Layer(Network):
    def __init__(self, layer_number, number_of_nodes, number_of_layers):
        # super().__init__(hyper_parameters)
        self.a = np.zeros(number_of_nodes)
        self.A = np.append(self.a, 1)
        self.weights = None
        self.D = np.zeros(number_of_nodes)
        self.delta = np.zeros(number_of_nodes)
        self.sigma = np.zeros(number_of_nodes)
        self.layer_number = layer_number
        self.non = number_of_nodes


def initialize_network():
        hyper_param = namedtuple('Hyper_Parameters', ['learning_rate',
                                                      'nodes_pr_layer',
                                                      'epochs'],)
        hyper_parameters = hyper_param(0.5,
                                       (5, 4, 3, 10),  # (In, Hidden, ..., Out)
                                       # (784, 16, 16, 10),  # (In, Hidden, ..., Out)
                                       1,)

        net = Network(hyper_parameters)
        net.load_data(5, [6, 7, 8, 9, 10])
        #
        # print('c', net.c)
        # print('blyf', net.layer[0].a)

        return net


def train_network(net):
    # Load training data
    # label_data = read_label_data('train-labels.idx1-ubyte')
    # picture_data = read_picture_data('train-images.idx3-ubyte')
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')
    net.load_data(label_data, picture_data) #TODO Pass in one data set at a time
    forward_pass()
    back_prop()
    update_weights()
    store_weights()


def evaluate_network(net):
    # Load evaluation data
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')
    load_weights()


if __name__ == '__main__':
    main()
