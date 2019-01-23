function [train_imgs,trian_labels,test_imgs,test_labels] = load_data()
test_imgs = load_images("../../mnist_data/t10k-images-idx3-ubyte");
test_labels = load_labels("../../mnist_data/t10k-labels-idx1-ubyte");
train_imgs = load_images("../../mnist_data/train-images-idx3-ubyte");
trian_labels = load_labels("../../mnist_data/train-labels-idx1-ubyte");
end

