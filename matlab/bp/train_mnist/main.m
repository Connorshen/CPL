clearvars
% hyperparameter
epoch = 1;
batch_size = 2000;
learning_rate = 0.1;
neurons = [784,10,10];
acts = [activation.none,activation.sigmoid,activation.softmax];
t_layer = [layer.dense,layer.dense,layer.dense];
params = init_params(epoch,batch_size,learning_rate,neurons,acts,t_layer);

% load data
[train_imgs,trian_labels,test_imgs,test_labels] = load_data();
%run training
[weights,biass,params,loss_all,batch_index_all]= run_training(train_imgs,trian_labels,test_imgs,test_labels,params);
% run testing
fprintf('accuracy = %.2f%%\n',run_testing(weights,biass,test_imgs,test_labels,params)*100);
% save_weights
save_weights(weights,biass,params);

% show batch loss
figure
plot(batch_index_all,loss_all);