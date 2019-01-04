%CNN�ٷ��̳�
clearvars
%ѵ����·��
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
figure;
%���������
perm = randperm(10000,20);
for i = 1:20
    %������ͼ
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

labelCount = countEachLabel(imds)
img = readimage(imds,1);
img_size = size(img)
%ÿ��750��ͼ����ѵ����250��������֤
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');
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
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%��ʼѵ��
net = trainNetwork(imdsTrain,layers,options);
%��֤׼ȷ��
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
