function accuracy=run_testing(test_imgs,test_labels,w1,w2,b1,b2)
test_img_size = size(test_imgs,1);
error = 0;
for i = 1:test_img_size
    label = test_labels(i,:);
    x = test_imgs(i,:)';
    % forward
    a1 = x;
    z2 = w1*a1+b1;
    a2 = sigmoid(z2);
    z3 = w2*a2+b2;
    %a3 = sigmoid(z3);
    out = softmax(z3);
    
    [~,id_l] = max(label);
    [~,id_o] = max(out);
    if id_l ~= id_o
        error = error +1;
    end
end
accuracy = (test_img_size-error)/test_img_size;
end