function train_bp()
data = train_data();
xs = data.xs;
ys = data.ys;
data_size = size(xs,1);
for i=1:init_params.epoch
    train_index = 1;
    while train_index<= data_size
        % get batch data
        batch_end = train_index+init_params.batch_size;
        if batch_end > data_size
            batch_end = data_size;
        end
        % forward
        for j = train_index:batch_end
            label = ys(j,:);
            x = xs(j,:)';
            
            input_size = size(x,1);
            middle_size = 5;
            out_size = 2;
            
            w1 = randn(middle_size,input_size);
            b1 = randn(middle_size,1);
            w2 = randn(out_size,middle_size);
            b2 = randn(out_size,1);
            
            a1 = x; 
            z2 = w1*a1+b1;
            a2 = sigmoid(z2);
            z3 = w2*a2+b2;
            a3 = sigmoid(z3);
            a3 = softmax(a3)
            
            
            
            train_index = train_index+1;
        end
        % backward
    end
end