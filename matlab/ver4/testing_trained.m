
clearvars
clc
load('results_trained.mat', 'results')

network_trained = results.network_trained;
init_para = results.init_para;

testing_result = run_testing( network_trained, init_para);