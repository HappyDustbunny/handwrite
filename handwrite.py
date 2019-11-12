import os
import numpy as np

def main():
    print('rap')
    label_data = read_label_data()
    picture_data = read_picture_data()
    print('blaf', picture_data)

def read_picture_data():
    file_name = os.path.join('.', 'datas', 'train-images.idx3-ubyte')

    with open(file_name, 'rb') as file:
        header1 = file.read(4)
        header2 = file.read(4)
        print(header1, header2)
        read_data = file.read()

    picture_data = np.zeros((60000, 784))  # 28*28 = 784

    s = 0
    for n in range(0, 60000*784, 784):
        for t, byte in enumerate(read_data[n: n + 784]):
            # print(f's {s} t {t} byte {byte}')
            picture_data[s, t] = byte
        s += 1
    print('blyf', picture_data)
    return picture_data

def read_label_data():
    file_name = os.path.join('.', 'datas', 'train-labels.idx1-ubyte')

    with open(file_name, 'rb') as file:
        header1 = file.read(4)
        header2 = file.read(4)
        print(header1, header2)
        read_data = file.read()

    label_data = np.zeros((60000, 784))  # 28*28 = 784

    s = 0
    for n in range(0, 60000*784, 784):
        for t, byte in enumerate(read_data[n: n + 784]):
            # print(f's {s} t {t} byte {byte}')
            label_data[s, t] = byte
        s += 1
    print('blyf', label_data)
    return label_data


if __name__ == '__main__':
    main()
