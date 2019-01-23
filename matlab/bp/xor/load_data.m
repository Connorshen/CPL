function [train_datas,train_labels,test_datas,test_labels] = load_data()
train_datas = [0,0;0,1;1,0;1,1];
train_labels = [0,1;1,0;1,0;0,1];
test_datas = [0,0;0,1;1,0;1,1];
test_labels = [0,1;1,0;1,0;0,1];
end