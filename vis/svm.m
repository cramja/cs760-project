% L1 SVM with L2 regularizer

x = create_seperable_data();
x = [ones(size(x, 1), 1) x]; % add bias term
y = x(:, 4);
x = x(:, 1:3);

clf;
fig = figure(1);
hold on;
set(fig, 'Position', [100 100 1000 800]);
sz = size(x);
scatter(x(1:sz(1)/2,2),x(1:sz(1)/2,3), 'r');
scatter(x(sz(1)/2:end,2),x(sz(1)/2:end,3), 'b');
xlabel('x_1');
ylabel('x_2');
xlim([-5 5]);
ylim([-5 5]);

epochs = 100;
learning_rate = 0.05;
learning_rate_decay = 0.95;
lambda = 1;

w = (rand(1,sz(2)) - 0.5) * 4;
w_h = [w;w];
for e = 1:epochs;
    u = 0;
    for i = randperm(sz(1));
        j = (x(i, :) * w') .* y(i) ;%.* -1;
        if j < 1
            u = u + 1;
            w = w - (learning_rate .* (x(i,:) .* y(i) .* -1 + (w .* lambda .* inv(sz(1)))));
        else % hinge
            w = w - (learning_rate .* (w .* lambda .* inv(sz(1))));
        end
        w_h = [w_h; w];
        
        if exist('w_scatter')
            delete(w_scatter);
        end
        if exist('w_boundry')
            delete(w_boundry);
        end
        boundary_x = [-4:0.25:4];
        boundary_y = (-1 .* ((w(2) .* boundary_x) + w(1))) ./  w(3);
        w_boundry = plot(boundary_x, boundary_y,'r');
        w_scatter = scatter(w_h(:,2), w_h(:,3), 'k.');
    end
    u
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x * w') == y) / sz(1);
    disp([e accuracy]);
    learning_rate = learning_rate * learning_rate_decay;
end