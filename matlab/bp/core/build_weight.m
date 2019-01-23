function [weights,biass,a_all,z_all,act_types,D_w_init,D_b_init] = build_weight(params)
n_layer = params.n_layer;
n_neurons = params.layers(:,3);
act_types = params.layers(:,2);

weights = cell(n_layer-1,1);
biass = cell(n_layer-1,1);
D_w_init = cell(n_layer-1,1);
D_b_init = cell(n_layer-1,1);
a_all = cell(n_layer,1);
for i =1:length(weights)
    weights{i,1} = randn(n_neurons{i+1},n_neurons{i});
    biass{i,1} = randn(n_neurons{i+1},1);
    D_w_init{i,1} = zeros(n_neurons{i+1},n_neurons{i});
    D_b_init{i,1} = zeros(n_neurons{i+1},1);
    
    a_all{i,1} = zeros(n_neurons{i},1);
end
a_all{n_layer,1} = zeros(n_neurons{n_layer},1);
z_all = a_all;
end