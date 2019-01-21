function [imgs, labels] = get_filterdata(digit_label, type)

labels = [];
imgs = [];

for label = digit_label
    filename = ['../../filterdata/' type int2str(label)];
    load(filename);
    
    for n = 1:size(D_filtered,1)
        labels(end+1) = label;
    end
    imgs = [imgs;D_filtered];
end

labels = labels';
num_digit_data = randperm(length(labels));
labels = labels(num_digit_data, :);
imgs = imgs(num_digit_data, :);