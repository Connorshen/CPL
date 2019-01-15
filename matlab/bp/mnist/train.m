clear all
[train_imgs,trian_labels,test_imgs,test_labels]=loadData();
xs = train_imgs;
ys = trian_labels;
data_size = size(xs,1);
middle_size = init_params.middle_size;
input_size = init_params.input_size;
out_size = init_params.out_size;
learning_rate = init_params.learning_rate;

w1 = randn(middle_size,input_size);
b1 = randn(middle_size,1);
w2 = randn(out_size,middle_size);
b2 = randn(out_size,1);
batchs = [];
all_loss = [];
train_index = 1;
batch_index = 1;
for i=1:init_params.epoch
    epoch_index = 1;
    while epoch_index<= data_size
        % get batch data
        batch_end = epoch_index+init_params.batch_size;
        if batch_end > data_size
            batch_end = data_size;
        end
        batch_size = batch_end - epoch_index+1;
        DW_1 = zeros(size(w1));
        DW_2 = zeros(size(w2));
        DB_1 = zeros(size(b1));
        DB_2 = zeros(size(b2));
        batch_loss = [];
        for j = epoch_index:batch_end
            label = ys(j,:)';
            x = xs(j,:)';
            % forward
            a1 = x;
            z2 = w1*a1+b1;
            a2 = sigmoid(z2);
            z3 = w2*a2+b2;
            %a3 = sigmoid(z3);
            out = softmax(z3);
            % backward
            h = out;
            deltaw_3 = h - label;
            deltaw_2 = w2'*deltaw_3.*sigmoid_gradient(z2);
            
            DW_2 = DW_2 + deltaw_3.*a2';
            DW_1 = DW_1 + deltaw_2.*a1';
            DB_2 = DB_2 + deltaw_3;
            DB_1 = DB_1 + deltaw_2;
            loss = -sum(label.*log(out));
            
            
            batch_loss(end+1)=loss;
            epoch_index = epoch_index+1;
            train_index = train_index+1;
        end
        w1_grad = (1.0 / batch_size) * DW_1;
        w2_grad = (1.0 / batch_size) * DW_2;
        b1_grad = (1.0 / batch_size) * DB_1;
        b2_grad = (1.0 / batch_size) * DB_2;
        w1 = w1-learning_rate*w1_grad;
        w2 = w2-learning_rate*w2_grad;
        b1 = b1-learning_rate*b1_grad;
        b2 = b2-learning_rate*b2_grad;
        
        all_loss(end+1)=mean(batch_loss);
        batchs(end+1)=batch_index;
        batch_index = batch_index+1;
    end
    fprintf('epoch = %d , acc = %.2f%%\n',i,run_testing(test_imgs,test_labels,w1,w2,b1,b2)*100);
end
figure
plot(batchs',all_loss')
title('batch loss')
xlabel('batch')
ylabel('loss')
