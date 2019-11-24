import os
import time
import numpy as np
import pygame

def main():
    print('Reading labels')
    label_data = read_label_data()

    print('Reading images')
    picture_data = read_picture_data()

    plot_stuff(picture_data, label_data)

def plot_stuff(picture_data, label_data):
    width, height = 28, 28  # Size of pixel when drawing numbers
    black = (0, 0, 0)
    pygame.init()
    screen = pygame.display.set_mode((700, 700))
    myfont = pygame.font.SysFont("monospace", 85)

    colour_data = picture_data[0: 784]

    n = 0
    while n < 784*20:
        screen.fill(black)  # Clear screen

        # Plot current number
        for y in range(0, 280, 10):
            for x in range(0, 280, 10):
                r = g = b = int(colour_data[int(n/784)][int(x/10) + int(28*y/10)])
                pygame.draw.rect(screen, (r, g, b), (x + 50, y + 50, 10, 10), 0)

        # Print target value for current number
        label = myfont.render(str(label_data[int(n/784)]), 1, (255,255,0))
        screen.blit(label, (400, 170))

        pygame.display.update()
        time.sleep(.05)

        n += 784


def read_picture_data():
    file_name = os.path.join('.', 'datas', 'train-images.idx3-ubyte')

    with open(file_name, 'rb') as file:
        read_data = file.read()

    picture_data = np.zeros((60000, 784))  # 28*28 = 784

    s = 0
    for n in range(16, 60000*784, 784):  # The first 16 bytes has to be dumped
        for t, byte in enumerate(read_data[n: n + 784]):
            picture_data[s, t] = byte
        s += 1

    return picture_data


def read_label_data():
    file_name = os.path.join('.', 'datas', 'train-labels.idx1-ubyte')

    with open(file_name, 'rb') as file:
        read_data = file.read()

    label_data = np.zeros((60000))  # 28*28 = 784

    for n in range(0, 60000):
        label_data[n] = read_data[n + 8]  # The first 8 bits has to be dumped

    return label_data


if __name__ == '__main__':
    main()
