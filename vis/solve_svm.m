function [ svm_state ] = solve_svm( method, iterations, svm_state )
%SOLVE_SVM Summary of this function goes here
%   method: which solver method to use:
%               - sgd
%               - asgd
%               - dual_sgd
%               - saga
%               - batch

if strcmp('sgd',method)
    svm_state = svm_sgd(iterations, svm_state)
end
    

end

function svm_sgd(svm_state)
    epochs = 50;
    learning_rate = 0.1;
    learning_rate_decay = 0.95;
    lambda = 1;

    w = (rand(1,sz(2)) - 0.5) * 30;
    for e = 1:epochs;
        for i = randperm(sz(1));
            j = (x(i, :) * w') .* y(i);
            if j < 1
                d_w = x(i,:) .* y(i) .* -1 + (w .* lambda .* inv(sz(1)));
            else % hinge
                d_w = w .* lambda .* inv(sz(1));
            end
            w = w - learning_rate * d_w;
            wvis = update_plot(w, wvis, 'r');
        end
        pause(0.05);
        accuracy = sum(sign(x * w') == y) / sz(1);
        disp([e accuracy]);
        learning_rate = learning_rate * learning_rate_decay;
    end
end