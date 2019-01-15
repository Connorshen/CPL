classdef init_params
    properties(Constant)
        epoch = 100;
        batch_size = 1;
        learning_rate = 0.01;
        
        mehod = train_method.type_bp;
        
        % for bp
        middle_size = 5;
        input_size = 2;
        out_size = 2;
    end
end
