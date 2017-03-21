% SVM with l2 regularizer using coordinate descent
% uses the dual problem formulation in its solver
%

[x y sz] = setup_plot();

epochs = 50;
learning_rate = 0.01;
learning_rate_decay = 0.95;
lambda = 1;

dual_plot = -1;

a = rand(sz(1),1) - 0.5; % [-0.5,0.5]
w = x' * a;
K = x * x'; % full linear kernel
for e = 1:epochs;
    % iterate for each a, coordinate descent
    j = randperm(sz(1));
    x = x(j, :);
    y = y(j);
    a = a(j);
    K = x * x';
    for i = randperm(sz(1))
        cls = (y .* (K*a)) < 1; % if 1, mis-classification
        da_i = (K(:,i) .* y .* -1)' * cls;
        reg = lambda * (K(i,:) * a);
        
        a(i) = a(i) - learning_rate * (da_i + reg);
        w = (x' * a);
        
        dual_plot = update_plot(w, dual_plot, 'r');
    end
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x * w) ~= y) / sz(1);
    disp([accuracy e]);
    learning_rate = learning_rate * learning_rate_decay;
end