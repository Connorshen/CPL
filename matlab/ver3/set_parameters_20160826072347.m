

function  init_para = set_parameters()

init_para.digit_label = [0:9]; 

init_para.numNeurons_input = 2560;
init_para.numNeurons_CPL = 100000;
init_para.numNeurons_cluster = 10;
init_para.numNeurons_decision = 10;

init_para.prob_input_CPL = 0.01;
init_para.hprob_input_CPL = [0.001:0.005:0.1];

init_para.num_rounds = 20;
init_para.trials_round = 500;
init_para.flag_plasticity_firstLayer = false;
init_para.flag_herg_inputCPL = false;

init_para.learning_rate = 0.5;
init_para.cond_decision = 3;
init_para.bound_decision_weight = 15000;
init_para.low_decision_weight = 5500;
init_para.high_decision_weight = 5502;
init_para.potential_prob = 0.95;
init_para.depress_prob = 0.45;
init_para.potential_gain = 1;
init_para.depress_gain = 0.1;





