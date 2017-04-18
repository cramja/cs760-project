% SVM with l2 regularizer using the SAGA update algorithm
%

[x y sz] = setup_plot();

epochs = 50;
learning_rate = 0.05;
learning_rate_decay = 0.95;
lambda = 0.5;

% used by the plot helper functions:
w_plot = -1;

w_saga = (rand(1,sz(2)) - 0.5) * 30;
d_wp = zeros(sz(1),3); % derivative from last iter

t = 0; % averaging time
for e = 1:epochs;
    for i = randperm(sz(1));
        t = t + 1;
        j = (x(i, :) * w_saga') .* y(i);
        if j < 1
            d_w = x(i,:) .* y(i) .* -1 + (w_saga .* lambda .* inv(sz(1)));
        else % hinge
            d_w = w_saga .* lambda .* inv(sz(1));
        end
        d_wsaga = d_w - d_wp(i,:) + (sum(d_wp)/min(sz(1), t));
        d_wp(i, :) = d_w;
        w_saga = w_saga - learning_rate * d_wsaga;
        
        w_plot = update_plot(w_saga, w_plot, 'r');
    end
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x * w_saga') == y) / sz(1);
    disp([e accuracy]);
    learning_rate = learning_rate * learning_rate_decay;
end