function y = gradient(x,type)
if type == activation.relu
    y = relu_gradient(x);
elseif type == activation.sigmoid
    y = sigmoid_gradient(x);
else
    y = x;
end