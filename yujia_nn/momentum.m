x1 = -1:0.01:1;
x2 = -1:0.01:1;
[X1,X2] = meshgrid(x1,x2);
Z = X1.^2 + (5*X2).^2;
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

% these are the positions at each time step
x1 = zeros(T,1);
x2 = x1;
x1(1) = -1;
x2(1) = -1;

x_m1 = x1;
x_m2 = x2;
v_m1 = 0; % ... and the velocity at t_i
v_m2 = 0;

x_m21 = x1; % ... and the momentum at t_i
x_m22 = x2;
v_m21 = 0;
v_m22 = 0;

x_m31 = x1;
x_m32 = x2;
v_m31 = 0;
v_m32 = 0;

scatter(0, 0, 200, 'r', 'filled');

for t = 1:T-1
    %sd contour
    sdPlot = plot(x1(1:t), x2(1:t), 'b', 'Linewidth', 3);


    x1(t+1) = x1(t) - eta * 2 * x1(t);
    x2(t+1) = x2(t) - eta * 50 * x2(t);
    
    %momentum
    mPlot = plot(x_m1(1:t), x_m2(1:t), 'm', 'Linewidth', 3);
    m2Plot = plot(x_m21(1:t), x_m22(1:t), 'g', 'Linewidth', 3);
    m3Plot = plot(x_m31(1:t), x_m32(1:t), 'c', 'Linewidth', 3);
    
    sdScatter = scatter(x1(t), x2(t),  100, 'b', 'filled');
    mScatter = scatter(x_m1(t), x_m2(t), 100, 'm', 'filled');
    m2Scatter = scatter(x_m21(t), x_m22(t), 100, 'g', 'filled');
    m3Scatter = scatter(x_m31(t), x_m32(t), 100, 'c', 'filled');
    
    v_m1 = 0.9*v_m1 - eta * 2 * x_m1(t);
    v_m2 = 0.9*v_m2 - eta * 50 * x_m2(t);
    x_m1(t+1) = x_m1(t) + v_m1;
    x_m2(t+1) = x_m2(t) + v_m2;
    
    v_m21 = 0.8*v_m21 - eta * 2 * x_m21(t);
    v_m22 = 0.8*v_m22 - eta * 50 * x_m22(t);
    x_m21(t+1) = x_m21(t) + v_m21;
    x_m22(t+1) = x_m22(t) + v_m22;
    
    v_m31 = 0.5*v_m31 - eta * 2 * x_m31(t);
    v_m32 = 0.5*v_m32 - eta * 50 * x_m32(t);
    x_m31(t+1) = x_m31(t) + v_m31;
    x_m32(t+1) = x_m32(t) + v_m32;
    
    legend('Contour','Optimal','Steepest Descent, eta=0.005', 'Momentum, mu=0.9, eta=0.005', 'Momentum, mu=0.8, eta=0.005', 'Momentum, mu=0.5, eta=0.005');
    pause(0.05);
    if (t == 1)
        pause(1);
    end
    delete(sdPlot);
    delete(mPlot);
    delete(m2Plot);
    delete(m3Plot);
    delete(sdScatter);
    delete(mScatter);
    delete(m2Scatter);
    delete(m3Scatter);
end
sdPlot = plot(x1, x2, 'b', 'Linewidth', 3);
mPlot = plot(x_m1, x_m2, 'm', 'Linewidth', 3);
m2Plot = plot(x_m21, x_m22, 'g', 'Linewidth', 3);
m3Plot = plot(x_m31, x_m32, 'c', 'Linewidth', 3);
sdScatter = scatter(x1(T), x2(T),  100, 'b', 'filled');
mScatter = scatter(x_m1(T), x_m2(T), 100, 'm', 'filled');
m2Scatter = scatter(x_m21(T), x_m22(T), 100, 'g', 'filled');
m3Scatter = scatter(x_m31(T), x_m32(T), 100, 'c', 'filled');