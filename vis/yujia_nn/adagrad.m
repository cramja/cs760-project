x1 = -1:0.01:1;
x2 = -1:0.01:1;
[X1,X2] = meshgrid(x1,x2);
Z = X1.^2+(5*X2).^2;
fig = figure(1);
set(fig, 'Position', [100 100 1400 1200])
contour(X1,X2,Z,100,'Linewidth',1.5)
set(gca,'fontsize',30)
xbounds = xlim;
set(gca,'XTick',xbounds(1):0.2:xbounds(2));
hold on
xlabel('w_1');
ylabel('w_2');

T = 200;
eta = 0.005;

x1 = zeros(T,1);
x2 = x1;
x1(1) = -1;
x2(1) = -1;

x_m1 = x1;x_m2 = x2;v_m1 = 0;v_m2 = 0;
x_n1 = x1;x_n2 = x2;v_n1 = 0;v_n2 = 0;
x_ada1 = x1;x_ada2 = x2; Gada1 = 0; Gada2 = 0;

scatter(0, 0, 200, 'r', 'filled');

for t = 1:T-1
    %sd
    sdPlot = plot(x1(1:t), x2(1:t), 'b', 'Linewidth', 3);
    mPlot = plot(x_m1(1:t), x_m2(1:t), 'm', 'Linewidth', 3);
    nPlot = plot(x_n1(1:t), x_n2(1:t), 'g', 'Linewidth', 3);
    adaPlot = plot(x_ada1(1:t), x_ada2(1:t), 'c', 'Linewidth', 3);
    
    sdScatter = scatter(x1(t), x2(t),  100, 'b', 'filled');
    mScatter = scatter(x_m1(t), x_m2(t), 100, 'm', 'filled');
    nScatter = scatter(x_n1(t), x_n2(t), 100, 'g', 'filled');
    adaScatter = scatter(x_ada1(t), x_ada2(t), 100, 'c', 'filled');
    
    x1(t+1) = x1(t) - eta * 2 * x1(t);
    x2(t+1) = x2(t) - eta * 50 * x2(t);
  
    v_m1 = 0.9*v_m1 - eta * 2 * x_m1(t);
    v_m2 = 0.9*v_m2 - eta * 50 * x_m2(t);
    x_m1(t+1) = x_m1(t) + v_m1;
    x_m2(t+1) = x_m2(t) + v_m2;
  
    v_n1 = 0.9*v_n1 - eta * 2 * (x_n1(t)+0.9*v_n1);
    v_n2 = 0.9*v_n2 - eta * 50 * (x_n2(t)+0.9*v_n2);
    x_n1(t+1) = x_n1(t) + v_n1;
    x_n2(t+1) = x_n2(t) + v_n2;
    
    Gada1 = Gada1 + (2 * (x_ada1(t))).^2;
    Gada2 = Gada2 + (50 * (x_ada2(t))).^2;
    x_ada1(t+1) = x_ada1(t) - 0.03 * (2 * (x_ada1(t)))/(sqrt(Gada1) + 1e-7);
    x_ada2(t+1) = x_ada2(t) - 0.03 * (50 * (x_ada2(t)))/(sqrt(Gada2) + 1e-7);
    
    
    legend('Contour','Optimal','Steepest Descent, eta=0.005', 'Momentum, mu=0.9, eta=0.005', 'Nesterov, mu=0.9, eta=0.005', 'Adagrad, eta=0.03');
    pause(0.005);
    if (t == 1)
        pause(1);
    end
    delete(sdPlot);
    delete(mPlot);
    delete(nPlot);
    delete(adaPlot);
    delete(sdScatter);
    delete(mScatter);
    delete(nScatter);
    delete(adaScatter);
end
sdPlot = plot(x1, x2, 'b', 'Linewidth', 3);
mPlot = plot(x_m1, x_m2, 'm', 'Linewidth', 3);
nPlot = plot(x_n1, x_n2, 'g', 'Linewidth', 3);
adaPlot = plot(x_ada1, x_ada2, 'c', 'Linewidth', 3);
sdScatter = scatter(x1(T), x2(T),  100, 'b', 'filled');
mScatter = scatter(x_m1(T), x_m2(T), 100, 'm', 'filled');
nScatter = scatter(x_n1(T), x_n2(T), 100, 'g', 'filled');
adaScatter = scatter(x_ada1(T), x_ada2(T), 100, 'c', 'filled');
