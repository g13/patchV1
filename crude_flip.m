function [xf, yf] = crude_flip(x,y,mx,my)
    coeff = pca([x' y']);
    k = coeff(1,1)/coeff(2,1);
    theta = atan2(coeff(2,1),coeff(1,1));
    sine2 = sin(theta)^2;
    sincos = sin(theta)*cos(theta);
    dx0 = (y-my)*k+mx - x; 
    dx = dx0*sine2;
    dy = dx0*sincos*sign(theta);
    xf = x + 2*dx;
    yf = y - 2*dy; 
end