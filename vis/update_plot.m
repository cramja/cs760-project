function [ wvis ] = update_plot( w, wvis_old, col )
%UPDATE_PLOT Plots a decision boundary given a weight vector
%   Assumes w is a 3 vector where w(1) is a bias weight, w(2) is x1 and
%   w(3) is x2
%
    % plot the decision boundary
    boundary_x = [-4:0.25:4];
    boundary_y = (-1 .* ((w(2) .* boundary_x) + w(1))) ./  w(3);
    line = plot(boundary_x, boundary_y, col);
    
    % plot the magnitude of the weight vector
    wv0 = [0, -w(1)/w(3)];
    m = ((-w(1) - w(2))/w(3)) - (-w(1)/w(3));
    ang = atan2(m, 1) + pi/2;
    nw = norm(w);
    wv1 = [((cos(ang) * nw) + wv0(1)) ((sin(ang) * nw + wv0(2)))];
    wv = [wv0; wv1];
    mag_line = plot(wv(:,1), wv(:,2), col);
    
    if isa(wvis_old, 'struct')
%         hist = [wv1 ; wvis_old.hist];
        delete(wvis_old.boundary);
        delete(wvis_old.mag_line);
%         delete(wvis_old.hist_scat);
    else
        hist = [wv1];
    end
    
%     hscat = scatter(hist(:,1), hist(:,2), 'k.');
    
    wvis.boundary = line;
    wvis.mag_line = mag_line;
%    wvis.hist_scat = hscat;
%    wvis.hist = hist;
end

