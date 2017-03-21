function [ x y sz ] = setup_plot( input_args )
%SETUP_PLOT returns training data
%   clears current plot and returns a fresh plot with new training data
%

    x = create_seperable_data();
    x = [ones(size(x, 1), 1) x]; % add bias term
    y = x(:, 4);
    x = x(:, 1:3);

    clf;
    fig = figure(1);
    hold on;
    set(fig, 'Position', [100 100 1000 800]);
    sz = size(x);
    scatter(x(1:sz(1)/2,2),x(1:sz(1)/2,3), 'r+');
    scatter(x(sz(1)/2:end,2),x(sz(1)/2:end,3), 'bo');
    xlabel('x_1');
    ylabel('x_2');
    xlim([-5 5]);
    ylim([-5 5]);
end

