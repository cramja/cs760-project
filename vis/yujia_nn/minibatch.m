% L = 1/40*((x-y1)^2+(x-y2)^2+...+(x+y39)^2+(x+y40)^2)
y = -19.5*5:5:19.5*5;
% batch
x = -100;
fig = figure(1);
set(fig, 'Position', [100 100 1400 1200])
eta = 0.2;
l = zeros(20,1);
for t = 1:20
    for i = y
        l(t) = l(t) + (x-i)^2;
    end
    l(t) = l(t)/40;
    grad = 0;
    for i = y
        grad = grad + 2*(x-i);
    end 
    grad = grad/40;
    x = x - eta*grad;
    % calculate loss
    
end
plot(0:40:760, l, 'Linewidth', 3);
hold on;
loss_batch = l(size(l,1)-10:end)

% SGD
x = -100;
l = zeros(800,1);
learning_rate = eta;
for t = 1:20 %epochs
    %shuffle
    ind = y;
    ind = ind(randperm(40));
    for i = 1:40
        %calculate loss
        tmp = (t-1)*40+i;
        for j = y
            l(tmp) = l(tmp) + (x-j)^2;
        end
        l(tmp) = l(tmp)/40;
        x = x - learning_rate * 2*(x-ind(i));
    end
    learning_rate = learning_rate * 0.8;
end
plot(0:799, l, 'Linewidth', 3)
loss_sgd = l(size(l,1)-10:end)

% mini batch
x = -100;
l = zeros(80,1);
learning_rate = eta;
for t = 1:20 %epochs
    %shuffle
    ind = y;
    ind = ind(randperm(40));
    for m = 1:4
        grad = 0;
        for i = 1:10
            grad = grad + 2*(x-ind((m-1)*10+i));
        end
        grad = grad / 10;
        %calculate loss
        tmp = (t-1)*4 + m;
        for j = y
            l(tmp) = l(tmp) + (x-j)^2;
        end
        l(tmp) = l(tmp)/40;  
        x = x - learning_rate*grad;
    end
    learning_rate = learning_rate * 0.9;
end
loss_mb = l(size(l,1)-10:end)

plot(0:10:790, l, 'Linewidth', 3)
set(gca,'fontsize',30)
xlabel('Num. of gradient evaluations');
ylabel('Loss');
legend('Batch gradient descent', 'Stochastic gradient descent', 'Mini-batch,size=10')
