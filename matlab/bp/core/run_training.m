function [weights,biass,params,loss_all,batch_index_all]= run_training(train_datas,train_labels,test_datas,test_labels,params)
xs = train_datas;
ys = train_labels;
data_size = size(xs,1);
learning_rate = params.learning_rate;
beta = params.beta;
loss_all = [];
batch_index_all = [];
[weights,biass,a_all,z_all,act_types,D_w_init,D_b_init] = build_weight(params);

batch_index = 1;
for i=1:params.epoch
    data_index = 1;
    V_w = D_w_init;
    V_b = D_b_init;
    while data_index<= data_size
        % get batch data
        batch_end = data_index+params.batch_size-1;
        if batch_end > data_size
            batch_end = data_size;
        end
        batch_start = data_index;
        batch_x = xs(batch_start:batch_end,:);
        batch_y = ys(batch_start:batch_end,:);
        batch_size = batch_end - batch_start+1;
        D_w = D_w_init;
        D_b = D_b_init;
  
        batch_loss = zeros(batch_size,1);
        for j = 1:batch_size
            x = batch_x(j,:)';
            y = batch_y(j,:)';
            
            % forward
            z_all{1,1} = x;
            a_all{1,1} = activate(z_all{1,1},act_types{1,1});
            for k = 1:length(weights)
                z_all{k+1,1} = weights{k,1}*a_all{k,1}+biass{k,1};
                a_all{k+1,1} = activate(z_all{k+1,1},act_types{k+1,1});
            end
            
            h = a_all{end,1};
            batch_loss(j)= -sum(y.*log(h));
            % backward
            delta = h - y;
            D_w{end,1} = D_w{end,1}+delta*a_all{end-1,1}';
            D_b{end,1} = D_b{end,1}+delta;
            for k = length(weights)-1:-1:1
                delta = weights{k+1,1}'*delta.*gradient(z_all{k+1,1},act_types{k+1,1});
                D_w{k,1} = D_w{k,1}+delta* a_all{k,1}';
                D_b{k,1} = D_b{k,1}+delta;
            end
            data_index = data_index+1;
        end 
        for j = 1:size(weights,1)
            d_w = D_w{j,1}/batch_size;
            d_b = D_b{j,1}/batch_size;
            V_w{j,1} = beta*V_w{j,1}-learning_rate*d_w;
            V_b{j,1} = beta*V_b{j,1}-learning_rate*d_b;
            weights{j,1} = weights{j,1}+V_w{j,1};
            biass{j,1} = biass{j,1}+V_b{j,1};
        end
        fprintf('epoch = %d bactch loss = %.2f batch_index = %d \n',i,mean(batch_loss),batch_index);
        
        loss_all(end+1)=mean(batch_loss);
        batch_index_all(end+1) = batch_index;
        batch_index = batch_index+1;
    end
end
