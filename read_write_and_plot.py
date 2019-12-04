import os
import time
import numpy as np
import pygame

os.environ['SDL_VIDEO_CENTERED'] = '1'  # Centers the Pygame window


def plot_stuff(single_label, single_picture, a_out):
    width, height = 10, 10  # Size of pixel when drawing numbers
    black, yellow = (0, 0, 0), (255, 255, 0)

    pygame.init()
    x, y = set_window_size()
    screen = pygame.display.set_mode((x, y))
    font_lbl = pygame.font.SysFont("monospace", 85)
    font_a = pygame.font.SysFont("monospace", 25)

    # colour_data = single_picture[0: 784]

    # n = 0
    # while n < 784*20:
    screen.fill(black)  # Clear screen

    # Plot current number
    for index, byte in enumerate(single_picture):
        r = g = b = int(byte)
        x = 5 * (index % 28 + 20)
        y = 5 * (int(index / 28) % 28 + 20)

        pygame.draw.rect(screen, (r, g, b), (x, y, width, height))

    # Print labels with 0, 1,..., 9 and colour them according to a_out
    for n in range(10):
        r = g = b = int(255 * a_out[0][n])
        a_label = font_a.render(str(n), 1, (r, g, b))
        screen.blit(a_label, (350, 30 * n + 20))

    # Print target value for current number
    label = font_lbl.render(str(single_label), 1, yellow)
    screen.blit(label, (470, 130))

    pygame.display.update()
    time.sleep(.05)

    # n += 784


def set_window_size():
    infostuffs = pygame.display.Info()  # gets monitor info

    monitorx, monitory = infostuffs.current_w, infostuffs.current_h  # puts monitor length and height into variables

    dispx, dispy = 700, 700

    if dispx > monitorx:  # scales screen down if too long
        dispy /= dispx / monitorx
        dispx = monitorx
    if dispy > monitory:  # scales screen down if too tall
        dispx /= dispy / monitory
        dispy = monitory

    x, y = int(dispx), int(dispy)

    return x, y


def read_picture_data(filename):
    """
    Reads a number of pictures with 28x28 pixels and stores them in numpy array
    """
    file_name = os.path.join('.', 'datas', filename)

    try:
        with open(file_name, 'rb') as file:
            read_data = file.read()
    except FileNotFoundError:
        print(f'Oups, the file {filename} was not found')

    try:
        if filename == 'train-images.idx3-ubyte':
            number_of_pics = 60000
        else:
            number_of_pics = 10000
    except LookupError:
        print(f'Oups, the file {filename} was not named as a MNist file')

    picture_data = np.zeros((number_of_pics, 28 * 28))  # 28*28 = 784

    s = 0
    for n in range(16, number_of_pics * 784, 784):  # 16 header bytes being dumped
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

    label_data = np.zeros(number_of_pics)

    for n in range(0, number_of_pics):
        label_data[n] = read_data[n + 8]  # 8 header bits being dumped

    print(f'\nLabel data read from {filename}')

    return label_data


def main():
    # label_data = read_label_data('train-labels.idx1-ubyte')
    # picture_data = read_picture_data('train-images.idx3-ubyte')
    label_data = read_label_data('t10k-labels.idx1-ubyte')
    picture_data = read_picture_data('t10k-images.idx3-ubyte')
    a_out = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    plot_stuff(picture_data, label_data, a_out)


if __name__ == '__main__':
    main()
