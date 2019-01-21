function accuracy = run_testing(weights,biass,test_imgs,test_labels,params)
[~,~,a_all,z_all,act_types,~,~] = build_weight(params);
n_error = 0;
test_img_length = length(test_imgs);
for i = 1:test_img_length
    x = test_imgs(i,:)';
    y = test_labels(i,:);
    
    % forward
    z_all{1,1} = x;
    a_all{1,1} = activate(z_all{1,1},act_types{1,1});
    for k = 1:length(weights)
        z_all{k+1,1} = weights{k,1}*a_all{k,1}+biass{k,1};
        a_all{k+1,1} = activate(z_all{k+1,1},act_types{k+1,1});
    end
    
    h = a_all{end,1};
    
    [~,ih]=max(h);
    [~,iy]=max(y);
    if ih~=iy
        n_error = n_error+1;
    end
end
accuracy =(test_img_length-n_error)/test_img_length;
end