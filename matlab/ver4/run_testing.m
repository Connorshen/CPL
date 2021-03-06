function testing_result = run_testing( network_trained, init_para)
disp('start testing')
% start time
tic
% get all training data set
disp('start loading data')
[ind_digit_data, digit_data] = get_filterdata(init_para.digit_label, 'test');
num_digit_data = size(ind_digit_data, 1);
testing_result = zeros(num_digit_data, 5);

for j = 1:num_digit_data
    
    label = ind_digit_data(j, 1);
    ind_label = ind_digit_data(j, 2);
    digit_img  = digit_data(ind_label,:)';

    input_CPL = network_trained.weight_input_CPL * digit_img;
    output_CPL = set_activity_CPL(input_CPL, network_trained.weight_recurrent_CPL, [init_para.numNeurons_CPL,init_para.numNeurons_cluster, init_para.flag_sparse, init_para.diff_th]);

    input_decision = network_trained.weightFilter_CPL_decision * output_CPL;
    prob_list_decision = exp(input_decision*init_para.gain_decision)./sum(exp(input_decision*init_para.gain_decision));
     
    [prob_decision, ind_decision] = max(prob_list_decision);  
    digit_decision = init_para.digit_label(ind_decision);
    if digit_decision == label
        reward = 1;
    else
        reward = 0;
    end
    
    testing_result(j, :) = [ind_label, label, digit_decision, reward, prob_decision];
    
    if mod(j,100) == 0
        fprintf("progress: %.2f%%\n",j/num_digit_data*100);
    end
end
end_time = toc;
fprintf('testing result: accuracy = %.2f%% cost_time = %.2fs\n',mean(testing_result(:, 4)*100),end_time);

    
    