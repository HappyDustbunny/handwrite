import os
import time
import numpy as np
import pygame


def plot_stuff(picture_data, label_data):
    width, height = 10, 10  # Size of pixel when drawing numbers
    black, yellow = (0, 0, 0), (255,255,0)
    pygame.init()
    screen = pygame.display.set_mode((700, 700))
    myfont = pygame.font.SysFont("monospace", 85)

    colour_data = picture_data[0: 784]

    n = 0
    while n < 784*20:
        screen.fill(black)  # Clear screen

        # Plot current number
        for y in range(0, 28*height, height):
            for x in range(0, 28*width, width):
                r = g = b = int(colour_data[int(n/784)][int(x/width) +
                                int(28*y/height)])
                pygame.draw.rect(screen, (r, g, b),
                                 (x + 50, y + 50, width, height), 0)

        # Print target value for current number
        label = myfont.render(str(label_data[int(n/784)]), 1, yellow)
        screen.blit(label, (400, 170))

        pygame.display.update()
        time.sleep(.05)

        n += 784


def read_picture_data(filename):
    """
    Reads a number of pictures with 28x28 pixels and stores them in numpy array
    """
    file_name = os.path.join('.', 'datas', filename)

    with open(file_name, 'rb') as file:
        read_data = file.read()

    try:
        if filename == 'train-images.idx3-ubyte':
            number_of_pics = 60000
        else:
            number_of_pics = 10000
    except FileNotFoundError:
        print(f'Oups, the file {filename} was not found')

    picture_data = np.zeros((number_of_pics, 28*28))  # 28*28 = 784

    s = 0
    for n in range(16, number_of_pics*784, 784):  # 16 header bytes being dumped
        for t, byte in enumerate(read_data[n: n + 784]):
            picture_data[s, t] = byte
        s += 1

    print(f'\nPicture data read from {filename}\n')

    return picture_data


def read_label_data(filename):
    file_name = os.path.join('.', 'datas', filename)

    with open(file_name, 'rb') as file:
        read_data = file.read()

    try:
        if filename == 'train-labels.idx1-ubyte':
            number_of_pics = 60000
        elif filename == 't10k-labels.idx1-ubyte':
            number_of_pics = 10000
    except FileNotFoundError:
        print(f'Oups, the file {filename} was not found')

    label_data = np.zeros((number_of_pics))

    for n in range(0, number_of_pics):
        label_data[n] = read_data[n + 8]  # 8 header bits being dumped

    print(f'\nLabel data read from {filename}')

    return label_data


def main():
    label_data = read_label_data('train-labels.idx1-ubyte')
    picture_data = read_picture_data('train-images.idx3-ubyte')
    plot_stuff(picture_data, label_data)


if __name__ == '__main__':
    main()