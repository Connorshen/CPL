clearvars
addpath(genpath('../core'))
[weights,biass,params] = load_weights();
[test_datas, test_labels] = get_filterdata(0:9, 'test');
accuracy = run_testing(weights,biass,test_datas,test_labels,params);
fprintf('accuracy = %.2f%%\n',accuracy*100);