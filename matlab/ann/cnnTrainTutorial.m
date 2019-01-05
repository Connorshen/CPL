%CNN官方教程
clearvars
%训练集路径
clearvars
[train_imgs,trian_labels,test_imgs,test_labels] = loadData();


%神经网络结构
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%配置超参数
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{test_imgs,test_labels}, ...
    'ValidationFrequency',100, ...
    'Verbose',true, ...
    'Plots','training-progress');
%开始训练
net = trainNetwork(train_imgs,trian_labels,layers,options);

%验证准确率
YPred = classify(net,test_imgs);
YValidation = test_labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

