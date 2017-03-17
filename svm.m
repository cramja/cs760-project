% SVM with l2 regularizer

x = create_seperable_data()

clf;
fig = figure(1);
hold on
set(fig, 'Position', [100 100 1000 800])
sz = size(x)
scatter(x(1:sz(1)/2,1),x(1:sz(1)/2,2), 'r')
scatter(x(sz(1)/2:end,1),x(sz(1)/2:end,2), 'b')
xlabel('x_1');
ylabel('x_2');
xlim([-5 5]);
ylim([-5 5]);

epochs = 50;
learning_rate = 0.01;
learning_rate_decay = 0.99;
lambda = 0.05;

w = (rand(1,2) - 0.5) * 4;
w_h = [w;w];
for e = 1:epochs
    x = x(randperm(sz(1)), :);
    for i = 1:sz(1)
        j = (x(i,1:2) * w') .* x(i,3);
        if j < 1
            w = w - learning_rate .* (x(i,1:2) .* x(i,3)) - (w .* lambda);
            w_h = [w_h; w];
        end
        if exist('w_scatter')
            delete(w_scatter);
        end
        if exist('w_boundry')
            delete(w_boundry);
        end
        a1 = [-4:0.25:4];
        a2 = (-1 .* w(1) .* a1) ./  w(2);
        w_boundry = plot(a1,a2,'r');
        w_scatter = scatter(w_h(:,1), w_h(:,2), 'k.');
    end
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x(:,1:2) * w') ~= x(:,3)) / sz(1)
    learning_rate = learning_rate * learning_rate_decay;
end