%CNN官方教程
clearvars
%训练集路径
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
%产生随机数
perm = randperm(10000,20);
for i = 1:20
    %绘制子图
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds)
img = readimage(imds,1);
img_size = size(img)
%每类750张图用来训练，250张用来验证
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
%神经网络结构
layers = [
    imageInputLayer([28 28 1])
    
    fullyConnectedLayer(20)
    reluLayer
    
    fullyConnectedLayer(20)
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%配置超参数
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%开始训练
net = trainNetwork(imdsTrain,layers,options);
%验证准确率
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
