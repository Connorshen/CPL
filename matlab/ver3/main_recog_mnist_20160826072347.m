% 这个版本主要是修改学习算法，突触要么连接，要么没有连接
% 2016.9.17

clear all
close all

rng('shuffle')

init_para = set_parameters();

network_init = initial_network(init_para);

[training_result, network_trained] = run_training( network_init,  init_para);

testing_result = run_testing( network_trained, init_para);