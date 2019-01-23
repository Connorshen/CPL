function gradient =  sigmoid_gradient(x)
gradient = sigmoid(x).*( 1 - sigmoid(x));