function labels = loadMNISTLabels(filename) 
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing 
%the labels for the MNIST images 
fp = fopen(filename, 'rb'); 
assert(fp ~= -1, ['Could not open ', filename, '']); 
magic = fread(fp, 1, 'int32', 0, 'ieee-be'); 
assert(magic == 2049, ['Bad magic number in ', filename, '']); 
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be'); 
ls = fread(fp, inf, 'unsigned char'); 
assert(size(ls,1) == numLabels, 'Mismatch in label count'); 
fclose(fp); 
label_num = size(ls,1);
labels = zeros(label_num,10);
for i = 1:label_num
    num = ls(i);
    labels(i,num+1)=1;
end
end
