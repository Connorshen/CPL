function train_bp()
data = train_data();
xs = data.xs;
ys = data.ys;
data_size = size(xs,1);
middle_size = init_params.middle_size;
input_size = init_params.input_size;
out_size = init_params.out_size;

w1 = randn(middle_size,input_size);
b1 = randn(middle_size,1);
w2 = randn(out_size,middle_size);
b2 = randn(out_size,1);
iterations = zeros(data_size*init_params.epoch,1);
all_loss = zeros(data_size*init_params.epoch,1);
train_index = 1;
for i=1:init_params.epoch
    epoch_index = 1;
    while epoch_index<= data_size
        % get batch data
        batch_end = epoch_index+init_params.batch_size;
        if batch_end > data_size
            batch_end = data_size;
        end
        batch_size = batch_end - epoch_index+1;
        D_1 = zeros(size(w1));
        D_2 = zeros(size(w2));
        for j = epoch_index:batch_end
            labels = ys(j,:)';
            x = xs(j,:)';
            % forward
            a1 = x;
            z2 = w1*a1+b1;
            a2 = sigmoid(z2);
            z3 = w2*a2+b2;
            a3 = sigmoid(z3);
            a3 = softmax(a3);
            % backward
            h = a3;
            deltaw_3 = h - labels;
            deltaw_2 = w2'*deltaw_3.*sigmoid_gradient(z2);
            
            D_2 = D_2 + deltaw_3.*a2';
            D_1 = D_1 + deltaw_2.*a1';
            loss = -sum(labels.*log(a3))
            
            iterations(train_index)=train_index;
            all_loss(train_index) = loss;
            epoch_index = epoch_index+1;
            train_index = train_index+1;
        end
        w1_grad = (1.0 / batch_size) * D_1;
        w2_grad = (1.0 / batch_size) * D_2;
        w1 = w1-w1_grad;
        w2 = w2-w2_grad;
    end
end
figure
plot(iterations,all_loss)
end