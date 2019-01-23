function save_weights(weights,biass,params)
result.weights = weights;
result.biass = biass;
result.params = params;
save('train_result','result');
end