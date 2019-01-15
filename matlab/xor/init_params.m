classdef init_params
    properties(Constant)
        epoch = 1000;
        batch_size = 4;
        learning_rate = 0.8;
        
        mehod = train_method.type_bp;
        
        % for bp
        middle_size = 5;
        input_size = 2;
        out_size = 2;
    end
end
