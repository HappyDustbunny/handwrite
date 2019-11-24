import numpy as np
from read_write_and_plot import read_label_data, read_picture_data, plot_stuff

def main():
    initialise_network()
    train_network()
    evaluate_network()


def train_network():
    # Load training data
    label_data = read_label_data('train-labels.idx1-ubyte')
    picture_data = read_picture_data('train-images.idx3-ubyte')
    forward_prop()
    back_prop()
    update_weights()
    store_weights()


def evaluate_network():
    # Load evaluation data
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')
    load_weights()

    
if __name__ == '__main__':
    main()
