import os
import scipy.io as sio
import numpy as np
from PIL import Image


def load_testdata(digit_label):
    data_ind = []
    data = []
    i_label = 1
    for label in digit_label:
        filename = os.path.join("../testdata", "test" + str(label) + ".mat")
        file_mat = sio.loadmat(filename)['D']
        num_digits = file_mat.shape[0]
        for i in range(num_digits):
            ind_correct = 0
            data_ind.append([label, i_label, ind_correct])
            data.append(file_mat[i])
            i_label += 1
    return np.array(data_ind), np.array(data)


def load_filterdata(digit_label, type):
    pass


# shape(784,)
def show_img(arr):
    print(arr.shape)
    img = Image.fromarray(arr.reshape(28, 28))
    img.show()


if __name__ == '__main__':
    ind, d = load_testdata([1, 2])
    show_img(d[0])
