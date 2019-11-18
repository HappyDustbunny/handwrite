import os
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    label_data = read_label_data()
    picture_data = read_picture_data()
    plot_stuff(picture_data, label_data)

def plot_stuff(picture_data, label_data):
    # colour = (.3, .80, .0)

    plt.style.use('dark_background')
    # plt.axis('off')
    # plt.grid(b=None)
    fig = plt.subplot()
    fig.axis('off')
    fig.set_xticks([])
    fig.set_yticks([])

    colour_data = picture_data[0: 784]
    print(len(colour_data), len(picture_data))
    for colour in colour_data:
        # fig.scatter(colour.reshape(28, 28), s=70, marker='s')  #, cmap='binary')
        plt.matshow(colour.reshape(28, 28))
        plt.show()
        print(label_data[0:5])
        time.sleep(1)
    # for yy in range(28):
    #     for xx in range(28):
    # n = 0
    # for number in colour_data:
    #     for col in number:
    #         colour = np.array([int(col), int(col), int(col)])
    #         x = n % 28
    #         y = int(n/28 % 28)
    #         n += 1
    #         # print('x, y, type(colour)', x, y, type(colour))
    #         print('x, y, colour, type(colour)', x, y, colour, type(colour))
    #         fig.scatter(x, 28 * y, s=70, c=colour, marker='s')  #, cmap='binary')
    #     plt.show()
    #         # https://ramdhaniverablog.wordpress.com/2016/09/30/
    #         # scatter-plot-and-color-mapping-in-python/
    # for yyy in range(0, 30, 3):
    #     fig.scatter(35, yyy, s=100, c=colour)
    # fig.set_aspect(1.0)
    # plt.show()

def read_picture_data():
    file_name = os.path.join('.', 'datas', 'train-images.idx3-ubyte')

    with open(file_name, 'rb') as file:
        header1 = file.read(4)
        header2 = file.read(4)

        read_data = file.read()

    picture_data = np.zeros((60000, 784))  # 28*28 = 784

    s = 0
    for n in range(0, 60000*784, 784):
        for t, byte in enumerate(read_data[n: n + 784]):
            # print(f's {s} t {t} byte {byte}')
            picture_data[s, t] = byte
        s += 1
    print('Image read')
    return picture_data

def read_label_data():
    file_name = os.path.join('.', 'datas', 'train-labels.idx1-ubyte')

    with open(file_name, 'rb') as file:
        header1 = file.read(4)
        header2 = file.read(4)

        read_data = file.read()

    label_data = np.zeros((60000, 784))  # 28*28 = 784

    s = 0
    for n in range(0, 60000*784, 784):
        for t, byte in enumerate(read_data[n: n + 784]):
            # print(f's {s} t {t} byte {byte}')
            label_data[s, t] = byte
        s += 1
    print('Labels read')
    return label_data


if __name__ == '__main__':
    main()
