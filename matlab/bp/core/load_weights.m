function [weights,biass,params] = load_weights()
load('train_result.mat')
weights = result.weights;
biass = result.biass;
params = result.params;
end