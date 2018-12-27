

function network = initial_network(init_para)

disp('initial network.')

% random group clusters in CPL，生成200000个元素打乱的序列
network.weight_recurrent_CPL = randperm(init_para.numNeurons_CPL);

% set weights between input and CLP
if init_para.flag_herg_inputCPL
    % 200000行
    row = init_para.numNeurons_CPL;
    % 2560列
    col = init_para.numNeurons_input;
    network.weight_input_CPL = zeros(row, col);
    % [0.001:0.001:0.01]，获取向量中元素的个数，为3
    num_prob = numel(init_para.hprob_input_CPL);
    % i=1,11,21,31..... 19991
    for i = 1:init_para.numNeurons_cluster:init_para.numNeurons_CPL
        % 簇内10个元素的下标索引
        ind_cluster = network.weight_recurrent_CPL(i:i+init_para.numNeurons_cluster-1);
        % 在三个元素随机取一个prob
        prob = init_para.hprob_input_CPL(randi(num_prob));
        % 对稀疏矩阵赋值
        for k = 1:init_para.numNeurons_cluster
            % sprand(m,n,density)生成一个m×n的服从均匀分布的随机稀疏矩阵，非零元素的分布密度是density
            % 想生成一个非均匀的稀疏矩阵？
            network.weight_input_CPL(ind_cluster(k), :) = sprand(1, col,prob);
        end
    end
else
    network.weight_input_CPL = full(sprand(init_para.numNeurons_CPL, init_para.numNeurons_input,...
                       init_para.prob_input_CPL));
end
% 大于0.5的置为1
network.weight_input_CPL(find(network.weight_input_CPL>0.5)) = 1;


% set the output weights
% r = randi([iMin,iMax],m,n)在开区间（iMin，iMax）生成m*n随机矩阵 shape:[10,200000]
network.weight_CPL_decision = randi([init_para.low_decision_weight, init_para.high_decision_weight],...
                                init_para.numNeurons_decision, init_para.numNeurons_CPL);
network.weightFilter_CPL_decision = tanh(network.weight_CPL_decision/init_para.bound_decision_weight);

