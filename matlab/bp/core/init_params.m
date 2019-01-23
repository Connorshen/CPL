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
        function params = init_params(epoch,batch_size,learning_rate,neurons,acts,t_layer)
            params.epoch = epoch;
            params.batch_size = batch_size;
            params.learning_rate = learning_rate;
            % it be used to compute momentum,beta is the discount of the before dw. eq:v_dW=βv_dW+(1-β)dW 
            params.beta = 0.9;
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
