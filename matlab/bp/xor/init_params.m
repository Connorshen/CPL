classdef init_params
    properties
        epoch
        batch_size
        learning_rate
        n_layer
        layers
    end
    methods
        function params = init_params()
            params.epoch = 1000;
            params.batch_size = 4;
            params.learning_rate = 0.1;
            % neuron of each layer
            neurons = [2 5 6 2];
            % activation of each layer
            acts = [activation.none,activation.relu,activation.relu,activation.softmax];
            % type of each layer
            t_layer = [layer.dense,layer.dense,layer.dense,layer.dense];
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
