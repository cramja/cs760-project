% L1 SVM with L2 regularizer. Uses the Averaging algorithm
% with gradient descent in the style of Bottou
%

[x y sz] = setup_plot();

epochs = 50;
learning_rate = 0.05;
learning_rate_decay = 0.95;
lambda = 0.5;

w_plot = -1;
w_avg_plot = -1;

w = (rand(1,sz(2)) - 0.5) * 30;
w_avg = w;

t = 0; % averaging time
for e = 1:epochs;
    for i = randperm(sz(1));
        t = t + 1;
        j = (x(i, :) * w') .* y(i);
        if j < 1
            d_w = x(i,:) .* y(i) .* -1 + (w .* lambda .* inv(sz(1)));
        else % hinge
            d_w = w .* lambda .* inv(sz(1));
        end
        w = ((1 - learning_rate * 0.2) * w) - (learning_rate * d_w); % convex combination
        if t > sz(1)
            mu = inv(t - sz(1));
        else
            mu = inv(t);
        end
        w_avg = w_avg + mu * (w - w_avg);
        
        w_avg_plot = update_plot(w_avg, w_avg_plot, 'k');
        w_plot = update_plot(w, w_plot, 'r');
    end
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x * w') == y) / sz(1);
    disp([e accuracy]);
    learning_rate = learning_rate * learning_rate_decay;
end