params = init_params;
[train_imgs,trian_labels,test_imgs,test_labels] = loadData();
[weights,biass,params,loss_all,batch_index_all]= run_training(train_imgs,trian_labels,test_imgs,test_labels,params);
save_weights(weights,biass,params);

accuracy = run_testing(weights,biass,test_imgs,test_labels,params)
% show batch loss
figure
plot(batch_index_all,loss_all);