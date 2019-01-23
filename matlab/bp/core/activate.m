function y = activate(x,type)
if type == activation.relu
    y = relu(x);
elseif type == activation.sigmoid
    y = sigmoid(x);
elseif type == activation.softmax
    y = softmax(x);
else
    y = x;
end
