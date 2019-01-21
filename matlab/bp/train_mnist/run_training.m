function [weights,biass,params,loss_all,batch_index_all]= run_training()
[train_imgs,trian_labels,test_imgs,test_labels] = loadData();
params = init_params;
xs = train_imgs;
ys = trian_labels;
data_size = size(xs,1);
learning_rate = params.learning_rate;
beta = params.beta;
loss_all = [];
batch_index_all = [];
[weights,biass,a_all,z_all,act_types,D_w_init,D_b_init] = build_weight(params);

batch_index = 1;
for i=1:params.epoch
    data_index = 1;
    D_w_before = D_w_init;
    D_b_before = D_b_init;
    while data_index<= data_size
        % get batch data
        batch_end = data_index+params.batch_size;
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
        % compute momentum
        for j=1:size(D_w,1)
            D_w{j,1} = D_w{j,1}+beta*D_w_before{j,1};
            D_b{j,1} = D_b{j,1}+beta*D_b_before{j,1};
        end
        % update weights
        for j = 1:size(weights,1)
            weights{j,1} = weights{j,1}-learning_rate*D_w{j,1}/batch_size;
            biass{j,1} = biass{j,1}-learning_rate*D_b{j,1}/batch_size;
        end
        accuracy = run_testing(weights,biass,test_imgs,test_labels,params)*100;
        fprintf('bactch loss = %.2f batch_index = %d accuracy = %.2f%%\n',mean(batch_loss),batch_index,accuracy);
        
        D_w_before = D_w;
        D_b_before = D_b;
        loss_all(end+1)=mean(batch_loss);
        batch_index_all(end+1) = batch_index;
        batch_index = batch_index+1;
    end
end