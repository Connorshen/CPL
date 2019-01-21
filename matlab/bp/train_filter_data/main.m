[weights,biass,params,loss_all,batch_index_all]= run_training();

% show batch loss
figure
plot(batch_index_all,loss_all);