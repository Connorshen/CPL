classdef train_data
    properties
        xs
        ys
    end
    methods
        function obj=train_data()
            % init data
            obj.xs = [0,0;0,1;1,0;1,1];
            obj.ys = [0;1;1;0];
           
            % upset
            randIndex = randperm(size(obj.xs,1));
            obj.xs = obj.xs(randIndex,:);
            obj.ys = obj.ys(randIndex,:);
        end
    end
end