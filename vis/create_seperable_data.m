function [ x ] = create_seperable_data()
%CREATE_SEPERABLE_DATA
%
%   returns matrix of the form [[x0,x1,y], [x0,x1,y], ... ]
%
    examples = 50;
    ex2 = examples/2;
    pd = makedist('Normal');
    x = [pd.random(examples, 2), ones(examples, 1)];
    x(1:ex2, 1:2) = x(1:examples/2, 1:2) + 1;
    x(ex2:examples, 1:2) = x(ex2:examples, 1:2) - 1;
    x(ex2:examples, 3) = -1;
end

