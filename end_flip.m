function [xf, yf, k] = end_flip(x,y,headOrTail, k)
    switch headOrTail
        case 'head'
            mx = x(1);
            my = y(1);
            kx = x(2)-x(1);
            ky = y(2)-y(1);
        case 'tail'
            mx = x(end);
            my = y(end);
            kx = x(end)-x(end-1);
            ky = y(end)-y(end-1);
    end
    if nargin < 4
        k = kx/ky;
        theta = atan2(ky,kx);
        if theta > pi/2
            theta = theta - pi;
        end
        if theta < -pi/2
            theta = theta + pi;
        end
    else
        theta = atan(1/k);
    end
    sine2 = sin(theta)^2;
    sincos = sin(theta)*cos(theta);
    dx0 = (y-my)*k+mx - x;
    dx = dx0*sine2;
    dy = dx0*sincos;
    xf = x + 2*dx;
    yf = y - 2*dy; 
end