function [train_imgs,trian_labels,test_imgs,test_labels] = loadData()
test_imgs = loadMNISTImages("../mnist_data/t10k-images-idx3-ubyte");
test_labels = loadMNISTLabels("../mnist_data/t10k-labels-idx1-ubyte");
train_imgs = loadMNISTImages("../mnist_data/train-images-idx3-ubyte");
trian_labels = loadMNISTLabels("../mnist_data/train-labels-idx1-ubyte");

figure;
%产生随机数
perm = randperm(60000,20);
for i = 1:20
    %绘制子图
    subplot(4,5,i);
    img = reshape(train_imgs(perm(i),:),28,28,1);
    imshow(img);
end