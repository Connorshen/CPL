clearvars
% hyperparameter
epoch = 1000;
batch_size = 4;
learning_rate = 0.5;
neurons = [2,10,2];
acts = [activation.none,activation.sigmoid,activation.softmax];
t_layer = [layer.dense,layer.dense,layer.dense];
params = init_params(epoch,batch_size,learning_rate,neurons,acts,t_layer);

% load data
[train_datas,train_labels,test_datas,test_labels] = load_data();
% run training
[weights,biass,params,loss_all,batch_index_all]= run_training(train_datas,train_labels,test_datas,test_labels,params);
% run testing
fprintf('accuracy = %.2f%%\n',run_testing(weights,biass,test_datas,test_labels,params)*100);
% save_weights
save_weights(weights,biass,params);
% show batch loss
figure
plot(batch_index_all,loss_all);
xlabel('iter');
ylabel('loss');