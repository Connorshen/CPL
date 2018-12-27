import os
import scipy.io as sio
import numpy as np
from PIL import Image
import h5py


def get_testdata(digit_label):
    print("start load test data:" + str(digit_label))
    data_ind = []
    data = []
    i_label = 0
    for label in digit_label:
        filename = os.path.join("../testdata", "test" + str(label) + ".mat")
        file_mat = sio.loadmat(filename)["D"]
        num_digits = file_mat.shape[0]
        for i in range(num_digits):
            ind_correct = 0
            data_ind.append([label, i_label, ind_correct])
            data.append(file_mat[i])
            i_label += 1
    print("successfully loaded test data,sum = " + str(i_label))
    return np.array(data_ind), np.array(data)


# type:"digit" or "test"
def get_filterdata(digit_label, type):
    print("start load filter data:" + str(digit_label))
    data_ind = []
    data = []
    i_label = 0
    for label in digit_label:
        filename = os.path.join("../filterdata", type + str(label) + ".mat")
        file_mat = h5py.File(filename)["D_filtered"]
        file_mat = np.array(np.transpose(file_mat))
        num_digits = file_mat.shape[0]
        for i in range(num_digits):
            ind_correct = 0
            data_ind.append([label, i_label, ind_correct])
            data.append(file_mat[i])
            i_label += 1
    np.random.shuffle(data_ind)
    print("successfully loaded filter data,sum = " + str(i_label))
    return np.array(data_ind), np.array(data)


# shape(784,)
def show_img(arr):
    print(arr.shape)
    img = Image.fromarray(arr.reshape(28, 28))
    img.show()


if __name__ == '__main__':
    test_ind, test_data = get_testdata([i for i in range(10)])
    # show_img(test_data[0])

    filter_ind, filter_data = get_filterdata([i for i in range(10)], "digit")
