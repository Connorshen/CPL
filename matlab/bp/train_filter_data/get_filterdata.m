function [imgs, labels] = get_filterdata(digit_label, type)

labels = [];
imgs = [];

for label = digit_label
    filename = ['../../filterdata/' type int2str(label)];
    load(filename);
    
    label_onehot = zeros(10,1);
    label_onehot(label+1) = 1;
    
    for n = 1:size(D_filtered,1)
        labels(end+1,:) = label_onehot;
    end
    imgs = [imgs;D_filtered];
end

num_digit_data = randperm(length(labels));
labels = labels(num_digit_data, :);
imgs = imgs(num_digit_data, :);