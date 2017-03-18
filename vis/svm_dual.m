% SVM with l2 regularizer using coordinate descent
% uses the dual problem formulation in its solver
%

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

epochs = 20;
learning_rate = 0.01;
learning_rate_decay = 0.90;
lambda = 1.0;

x = x(randperm(sz(1)), :);
a = rand(sz(1),1) - 0.5; % [-0.5,0.5]
w_h = x(:,1:2)' * a;
K = x(:,1:2) * x(:,1:2)'; % full linear kernel
% liblinear folks say that Q_i,j = y_i * y_j * (x_i' x_j)
% yy = repmat(x(:,3), 1, size(x,1)) .* repmat(x(:,3), 1, size(x,1))';
% K = K .* yy;
for e = 1:epochs
    % iterate for each a, coordinate descent
    j = randperm(sz(1));
    x = x(j, :);
    a = a(j);
    K = x(:,1:2) * x(:,1:2)';
    for i = randperm(sz(1))
        cls = (x(:,3) .* (K*a)) < 1; % if 1, mis-classification
        da_i = (K(:,i) .* x(:,3) .* -1)' * cls;
        %da_i = (x(:,3) .* K(:,i) .* a(i))' * cls;
        reg = lambda * (K(i,:) * a);
        
        a(i) = a(i) - learning_rate * (da_i + reg);
        w_hp = (x(:,1:2)' * a);
        w_h = [w_h w_hp];
        if exist('w_scatter', 'var')
            delete(w_scatter);
        end
        if exist('w_boundry', 'var')
            delete(w_boundry);
        end
        a1 = [-4:0.25:4];
        a2 = (-1 .* w_hp(1) .* a1) ./  w_hp(2);
        w_boundry = plot(a1,a2,'r');
        w_scatter = scatter(w_h(1,:)', w_h(2,:)', 'k.');
    end
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x(:,1:2) * w_hp) ~= x(:,3)) / sz(1);
    disp([accuracy e]);
    learning_rate = learning_rate * learning_rate_decay;
end

epochs = 50;
learning_rate = 0.01;
learning_rate_decay = 0.9;
lambda = 0.2;

w = (rand(1,2) - 0.5) * 4;
w_h = [w;w];
for e = 1:epochs
    x = x(randperm(sz(1)), :);
    for i = 1:sz(1)
        j = (x(i,1:2) * w') .* x(i,3);
        if j < 1
            w = w - learning_rate .* (x(i,1:2) .* x(i,3)) - (w .* lambda .* inv(sz(1)));
            w_h = [w_h; w];
        end
        if exist('w_scatter')
            delete(w_scatter);
        end
        if exist('w_boundry2')
            delete(w_boundry2);
        end
        a1 = [-4:0.25:4];
        a2 = (-1 .* w(1) .* a1) ./  w(2);
        w_boundry2 = plot(a1,a2,'r');
        w_scatter = scatter(w_h(:,1), w_h(:,2), 'k.');
    end
    pause(0.05); % allows plot to update
    accuracy = sum(sign(x(:,1:2) * w') ~= x(:,3)) / sz(1)
    learning_rate = learning_rate * learning_rate_decay;
end