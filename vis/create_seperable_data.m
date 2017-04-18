function [ x ] = create_seperable_data()
%CREATE_SEPERABLE_DATA
%
%   returns matrix of the form [[x0,x1,y], [x0,x1,y], ... ]
%   centers one dataset around 0,0 and the other around 3,3
%
    rng('default')
    examples = 50;
    ex2 = examples/2;
    pd = makedist('Normal');
    x = [pd.random(examples, 2) - 0.5, ones(examples, 1)];
    x(1:ex2, 1:2) = x(1:examples/2, 1:2) + 1;
    x(ex2:examples, 1:2) = x(ex2:examples, 1:2) + 3;
    x(ex2:examples, 3) = -1;
    
    x = x - repmat(sum(x)/examples, examples, 1); % center around 0,0
end

