classdef init_params
    properties
        epoch
        batch_size
        learning_rate
        beta
        n_layer
        layers
    end
    methods
        function params = init_params()
            params.epoch = 1000;
            params.batch_size = 200;
            params.learning_rate = 0.01;
            % it be used to compute momentum,beta is the discount of the before dw. eq:v_dW=βv_dW+(1-β)dW 
            params.beta = 0.9;
            % neuron of each layer
            neurons = [784 20 10];
            % activation of each layer
            acts = [activation.none,activation.sigmoid,activation.softmax];
            % type of each layer
            t_layer = [layer.dense,layer.dense,layer.dense];
            %num of layer
            params.n_layer = length(neurons);
            % combine layers to struct
            params.layers=cell(params.n_layer,3);
            for i = 1:params.n_layer
                params.layers(i,:)={t_layer(i),acts(i),neurons(i)};
            end
            params.print_struct();
        end
        function print_struct(self)
            fprintf('the structure of the neuron network:\n\n');
            for i = 1:self.n_layer
                disp(self.layers(i,:));
            end
        end
    end
end
