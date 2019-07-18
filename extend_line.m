function [xf, yf, k] = extend_line(x,y,headOrTail,ending)
    switch headOrTail
        case 'head'
            xs = x(1);
            ys = y(1);
            kx = x(1)-x(2);
            ky = y(1)-y(2);
        case 'tail'
            xs = x(end);
            ys = y(end);
            kx = x(end)-x(end-1);
            ky = y(end)-y(end-1);
    end
    if isfield(ending,'k')
        k = ending.k;
    else
        k = ky/kx;
    end
    switch ending.dim
        case 'x'
            n = round(abs((ending.bound-xs)/kx));
            if ending.bound-xs > 0
                xf = linspace(xs, ending.bound, n);
            else
                xf = fliplr(linspace(ending.bound, xs, n));
            end
            yf = k*(x-xs) + ys;
        case 'y'
            n = round(abs((ending.bound-ys)/ky));
            if ending.bound-xs > 0
                yf = linspace(ys, ending.bound, n);
            else
                yf = fliplr(linspace(ending.bound, ys, n));
            end
            xf = (yf-ys)/k + xs;
    end
end