%CNN�ٷ��̳�
clearvars
%ѵ����·��
[train_imgs,trian_labels,test_imgs,test_labels] = loadData();

%������ṹ
layers = [
    imageInputLayer([28 28 1])
    
    fullyConnectedLayer(20)
    reluLayer
    
    fullyConnectedLayer(20)
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
%���ó�����
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{test_imgs,test_labels}, ...
    'ValidationFrequency',100, ...
    'Verbose',true, ...
    'Plots','training-progress');
%��ʼѵ��
net = trainNetwork(train_imgs,trian_labels,layers,options);
%��֤׼ȷ��
YPred = classify(net,test_imgs);
YValidation = test_labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
