function save_model(network_trained,init_para)
results.network_trained = network_trained;
results.init_para = init_para;
save('results_trained.mat', 'results','-v7.3');
disp('successfully save model')