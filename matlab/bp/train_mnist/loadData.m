function [train_imgs,trian_labels,test_imgs,test_labels] = loadData()
test_imgs = loadMNISTImages("../../mnist_data/t10k-images-idx3-ubyte");
test_labels = loadMNISTLabels("../../mnist_data/t10k-labels-idx1-ubyte");
train_imgs = loadMNISTImages("../../mnist_data/train-images-idx3-ubyte");
trian_labels = loadMNISTLabels("../../mnist_data/train-labels-idx1-ubyte");
end

