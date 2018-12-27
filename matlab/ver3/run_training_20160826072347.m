
function [training_result, network_trained] = run_training( network_init,  init_para)

training_result = [];

% get all training data set
[ind_digit_data, digit_data] = get_data(init_para.digit_label, 'digit');
num_digit_data = size(ind_digit_data, 1);

for i = 1:init_para.num_rounds
    disp(' th round training...')
    disp(i)
    num_trials = init_para.trials_round;
    % prepare training set for each round
    ind_training_data = ind_digit_data(randi(num_digit_data,num_trials, 1), :);
    
    result_round = zeros(num_trials,4);
    
    for j = 1:num_trials
        label = ind_training_data(i, 1);
        ind_label = ind_training_data(i, 2);
        digit_img  = digit_data(ind_label,:)';
        
        input_CPL = network_init.weight_input_CPL * digit_img;
        output_CPL = set_activity_CPL(input_CPL, network_init.weight_recurrent_CPL, init_para.numNeurons_CPL,...
                                        init_para.numNeurons_cluster);
                                    
        input_decision = network_init.weightFilter_CPL_decision * output_CPL;
        mean_input_decision = input_decision - mean(input_decision);
        prob_list_decision = 1./(1+exp(-mean_input_decision.*init_para.cond_decision));
        
        [prob_decision, ind_decision] = max(prob_list_decision);  
        digit_decision = ind_decision - 1;
        if digit_decision == label
            reward = 1;
        else
            reward = 0;
        end
        
        % update the weights of the final layer
        
        wm = network_init.weight_CPL_decision(ind_decision, :);  % which synapses will be updated
        num_wm = numel(wm);
        
        val_potential = output_CPL'.* (rand(1,num_wm)<init_para.potential_prob);
        val_depress = ~output_CPL'.* (rand(1,num_wm)<init_para.depress_prob);
        val_potential = val_potential*init_para.potential_gain;
        val_depress = val_depress*init_para.depress_gain;
        
        if reward
            wm = wm + (val_potential - val_depress)*(reward - prob_decision);
            ind_digit_data(ind_label, 3) = 1;
        else
            wm = wm - val_potential;
        end
        % all weights need to be greater than 0
        wm = max(wm, 0);
        network_init.weight_CPL_decision(ind_decision, :) = wm;
        network_init.weightFilter_CPL_decision(ind_decision, :) = tanh(wm/init_para.bound_decision_weight);
        
        result_round(end+1, :) = [label, digit_decision, reward, prob_decision];
    end
    disp('training result in a round...');
    disp(mean(result_round(:, 3:4)));
    
    training_result = [training_result; result_round];   
end

network_trained = network_init;


