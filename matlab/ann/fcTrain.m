%CNN官方教程
clearvars
%训练集路径
[train_imgs,trian_labels,test_imgs,test_labels] = loadData();

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
    'MaxEpochs',10, ...
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
