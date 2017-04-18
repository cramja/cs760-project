classdef svm_sgd < handle
    %SVM_SGD Summary of this class goes here
    %   Detailed explanation goes here
    
    properties (Access = public)
        learning_rate
        learning_rate_decay
        lambda
        idx
        i
        epoch
        w
        sz
        x
        y
    end
    
    methods
        function obj = svm_sgd(x,y, learning_r, lambda)
            obj.x = x;
            obj.y = y;
            obj.learning_rate = learning_r;
            obj.learning_rate_decay = 0.9;
            obj.lambda = lambda;
            obj.sz = size(x);
            obj.w = (rand(1,obj.sz(2)) - 0.5) * 30;
            obj.epoch = 0;
            obj.i = 0;
            obj.idx = randperm(obj.sz(1));
        end

        function w = next(obj)
            if obj.i == obj.sz(1)
                obj.idx = randperm(obj.sz(1));
                obj.epoch = obj.epoch + 1;
                obj.i = 0;
                obj.learning_rate = obj.learning_rate * obj.learning_rate_decay;
            end
            obj.i = obj.i + 1;
            j = obj.idx(obj.i);
            
            loss = (obj.x(j, :) * obj.w') * obj.y(j);
            if loss < 1
                d_w = obj.x(j,:) .* obj.y(j) .* -1 + (obj.w .* obj.lambda .* inv(obj.sz(1)));
            else % hinge
                d_w = obj.w .* obj.lambda .* inv(obj.sz(1));
            end
            obj.w = obj.w - obj.learning_rate * d_w;
            
            w = obj.w;
        end

        function reset(obj)
            obj.w = (rand(1,obj.sz(2)) - 0.5) * 30;
            obj.epoch = 0;
            obj.i = 0;
            obj.idx = randperm(obj.sz(1));
        end
    end
end

