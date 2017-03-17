x = -0.5:0.01:0.5;
fig = figure(1);
set(fig, 'Position', [100 100 1400 1200])

plot(x, x.^2,'Linewidth', 3);
hold on
set(gca,'fontsize',30)
xlabel('w');
ylabel('Loss');

sd = -0.5;
ada = sd; adaG = 0;
rms = sd; rmsG = 0;
mom = sd; momV = 0;
nes = sd; nesV = 0;
eta = 0.01;
T = 120;
for t = 1:T-1
%     sdScatter = scatter(sd, sd^2,  200, 'b', 'filled');
%     nScatter = scatter(nes, nes^2, 200, 'g', 'filled');
    mScatter = scatter(mom, mom^2, 200, 'm', 'filled');
    adaScatter = scatter(ada, ada^2, 200, 'g', 'filled');
    rmsScatter = scatter(rms, rms^2, 200, 'b', 'filled');
    legend('Loss function','Momentum, mu=0.9, eta=0.01','Adagrad, eta=0.01', 'RMSProp, eta=0.01, gamma=0.9');

    sd = sd - eta*(2*sd);
    momV = 0.9*momV - eta * 2 * mom;
    mom = mom + momV;
    nesV = 0.9*nesV - eta * 2 * (nes+0.9*nesV);
    nes = nes + nesV;
    adaG = adaG + (2*ada)^2;
    ada = ada - eta*(2*ada)/(sqrt(adaG)+1e-7);
    rmsG = 0.9*rmsG + 0.1*(2*rms)^2;
    rms = rms - eta*(2*rms)/(sqrt(rmsG)+1e-7);
    pause(0.05);
    if (t == 1)
        pause(1);
    end
    delete(adaScatter)
    delete(mScatter)
%     delete(nScatter)
    delete(rmsScatter)
%     delete(sdScatter)
end
mScatter = scatter(mom, mom^2, 200, 'm', 'filled');
adaScatter = scatter(ada, ada^2, 200, 'g', 'filled');
rmsScatter = scatter(rms, rms^2, 200, 'b', 'filled');