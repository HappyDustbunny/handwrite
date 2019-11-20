import os
import time
import numpy as np
import pygame

def main():
    label_data = read_label_data()
    picture_data = read_picture_data()
    plot_stuff(picture_data, label_data)

def plot_stuff(picture_data, label_data):
    width, height = 28, 28
    black = (0, 0, 0)
    pygame.init()
    screen = pygame.display.set_mode((700, 700))

    n = 0
    colour_data = picture_data[0: 784]


    #print(colour_data, '\n', colour_data[[5]], '\n', colour_data[5][5])
    while n < 784*50:
        #colour_data = picture_data[0: n + 784]
        screen.fill(black)
        for y in range(0, 280, 10):
            for x in range(0, 280, 10):
                r = g = b = int(colour_data[int(n/784)][int(x/10) + int(28*y/10)])
                pygame.draw.rect(screen, (r, g, b), (x + 50, y + 50, 10, 10), 0)
        print('Label ', label_data[int(n/784)])
        pygame.display.update()
        #time.sleep(.5)

        n += 784


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

    label_data = np.zeros((60000))  # 28*28 = 784

    for n in range(0, 60000):
        #for t, byte in enumerate(read_data[n]):
            # print(f's {s} t {t} byte {byte}')
        label_data[n] = read_data[n]
    print('Labels read')
    return label_data


if __name__ == '__main__':
    main()
