clearvars
[weights,biass,params] = load_weights();
[~,~,test_datas,test_labels] = load_data();
accuracy = run_testing(weights,biass,test_datas,test_labels,params);
fprintf('accuracy = %.2f%%\n',accuracy*100);