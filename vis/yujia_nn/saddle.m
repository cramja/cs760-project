x1 = -1:0.01:1;
x2 = -1:0.01:1;
[X1,X2] = meshgrid(x1,x2);
Z = X1.^3+X2.^2;
fig = figure(1);
set(fig, 'Position', [0 -500 1400 1200])
contour(X1,X2,Z,100,'Linewidth',1.5)
set(gca,'fontsize',30)
xbounds = xlim;
set(gca,'XTick',xbounds(1):0.2:xbounds(2));
hold on
xlabel('w_1');
ylabel('w_2');

T = 150;
eta = 0.005;

x1 = zeros(T,1);
x2 = x1;
x1(1) = -0.01;
x2(1) = -1;

x_m1 = x1;x_m2 = x2;v_m1 = 0;v_m2 = 0;
x_n1 = x1;x_n2 = x2;v_n1 = 0;v_n2 = 0;
x_ada1 = x1;x_ada2 = x2; Gada1 = 0; Gada2 = 0;
x_rms1 = x1;x_rms2 = x2; Grms1 = 0; Grms2 = 0;
x_adam1 = x1;x_adam2 = x2; madam1 = 0; madam2 = 0; vadam1 = 0; vadam2 = 0;b1 = 0.9; b2=0.999;


scatter(0, 0, 200, 'r', 'filled');
stop = 0;
for t = 1:T-1
    if stop == 1
        break
    end
    %sd
    sdPlot = plot(x1(1:t), x2(1:t), 'b', 'Linewidth', 3);
    mPlot = plot(x_m1(1:t), x_m2(1:t), 'm', 'Linewidth', 3);
    nPlot = plot(x_n1(1:t), x_n2(1:t), 'g', 'Linewidth', 3);
    adaPlot = plot(x_ada1(1:t), x_ada2(1:t), 'c', 'Linewidth', 3);
    rmsPlot = plot(x_rms1(1:t), x_rms2(1:t), 'Color', [0.1, 0.3, 0.4] , 'Linewidth', 3);
    adamPlot = plot(x_adam1(1:t), x_adam2(1:t), 'Color', [0.4, 0.3, 0.1], 'Linewidth', 3);
  
    sdScatter = scatter(x1(t), x2(t),  100, 'b', 'filled');
    mScatter = scatter(x_m1(t), x_m2(t), 100, 'm', 'filled');
    nScatter = scatter(x_n1(t), x_n2(t), 100, 'g', 'filled');
    adaScatter = scatter(x_ada1(t), x_ada2(t), 100, 'c', 'filled');
    rmsScatter = scatter(x_rms1(t), x_rms2(t),  100,  [0.1, 0.3, 0.4] , 'filled');
    adamScatter = scatter(x_adam1(t), x_adam2(t), 100, [0.4 0.3 0.1], 'filled');

    axis([-1 1 -1 1]);
    
    x1(t+1) = x1(t) - eta * 3 * x1(t).^2;
    x2(t+1) = x2(t) - eta * 2 * x2(t);
  
    v_m1 = 0.9*v_m1 - eta * 3 * x_m1(t).^2;
    v_m2 = 0.9*v_m2 - eta * 2 * x_m2(t);
    x_m1(t+1) = x_m1(t) + v_m1;
    x_m2(t+1) = x_m2(t) + v_m2;
  
    v_n1 = 0.9*v_n1 - eta * 3 * (x_n1(t)+0.9*v_n1).^2;
    v_n2 = 0.9*v_n2 - eta * 2 * (x_n2(t)+0.9*v_n2);
    x_n1(t+1) = x_n1(t) + v_n1;
    x_n2(t+1) = x_n2(t) + v_n2;
    
    Gada1 = Gada1 + (3 * (x_ada1(t)).^2).^2;
    Gada2 = Gada2 + (2 * (x_ada2(t))).^2;
    x_ada1(t+1) = x_ada1(t) - 0.03 * (3 * (x_ada1(t)).^2)/(sqrt(Gada1) + 1e-7);
    x_ada2(t+1) = x_ada2(t) - 0.03 * (2 * (x_ada2(t)))/(sqrt(Gada2) + 1e-7);
    
    Grms1 = 0.9*Grms1 + 0.1*(3 * (x_rms1(t)).^2).^2;
    Grms2 = 0.9*Grms2 + 0.1*(2 * (x_rms2(t))).^2;
    x_rms1(t+1) = x_rms1(t) - 0.03 * (3 * (x_rms1(t)).^2)/(sqrt(Grms1) + 1e-7);
    x_rms2(t+1) = x_rms2(t) - 0.03 * (2 * (x_rms2(t)))/(sqrt(Grms2) + 1e-7);    
    
    madam1 = 0.9*madam1 + 0.1*(3 * (x_adam1(t)).^2);
    madam2 = 0.9*madam2 + 0.1*(2 * (x_adam2(t)));
    vadam1 = 0.999*vadam1 + 0.001*(3 * (x_adam1(t))^2).^2;
    vadam2 = 0.999*vadam2 + 0.001*(2 * (x_adam2(t)))^2;
    mhat1 = madam1/(1-b1);
    mhat2 = madam2/(1-b1);
    vhat1 = vadam1/(1-b2);
    vhat2 = vadam2/(1-b2);
    x_adam1(t+1) = x_adam1(t) - 0.03*mhat1/(sqrt(vhat1)+1e-8);
    x_adam2(t+1) = x_adam2(t) - 0.03*mhat2/(sqrt(vhat2)+1e-8);
    b1 = 0.9*b1;
    b2 = 0.999*b2;
    
    
    legend('Contour','Saddle Point','Steepest Descent, eta=0.005', 'Momentum, mu=0.9, eta=0.005', 'Nesterov, mu=0.9, eta=0.005',  'RMSProp, eta=0.03, gamma=0.9', 'AdaGrad, eta=0.03', 'Adam, eta=0.03, b1=0.9, b2=0.999');
    pause(0.005);
    if (t == 1)
        pause(1);
    end
    delete(sdPlot);
    delete(mPlot);
    delete(nPlot);
    delete(rmsPlot);
    delete(adaPlot);
    delete(adamPlot);
    delete(sdScatter);
    delete(mScatter);
    delete(nScatter);
    delete(adamScatter);
    delete(rmsScatter);
    delete(adaScatter);
end
sdPlot = plot(x1, x2, 'b', 'Linewidth', 3);
mPlot = plot(x_m1, x_m2, 'm', 'Linewidth', 3);
nPlot = plot(x_n1, x_n2, 'g', 'Linewidth', 3);
adaPlot = plot(x_ada1, x_ada2, 'c', 'Linewidth', 3);
rmsPlot = plot(x_rms1, x_rms2, 'Color', [0.1, 0.3, 0.4] , 'Linewidth', 3);
adamPlot = plot(x_adam1, x_adam2, 'Color', [0.4, 0.3, 0.1], 'Linewidth', 3);

sdScatter = scatter(x1(T), x2(T),  100, 'b', 'filled');
mScatter = scatter(x_m1(T), x_m2(T), 100, 'm', 'filled');
nScatter = scatter(x_n1(T), x_n2(T), 100, 'g', 'filled');
adaScatter = scatter(x_ada1(T), x_ada2(T), 100, 'c', 'filled');
rmsScatter = scatter(x_rms1(T), x_rms2(T),  100,  [0.1, 0.3, 0.4] , 'filled');
adamScatter = scatter(x_adam1(T), x_adam2(T), 100, [0.4 0.3 0.1], 'filled');