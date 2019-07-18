function [Pi, W, LR] = myCortex(G, ecc, a, b, k, resol, nod, rOD, noise, getLR, plot_patch)
    if nargin < 11
        plot_patch = 0;
        if nargin < 10
            getLR = true;
        end
    end
    nx = G(1);
    ny = G(2);
    Pi = ones(nx,ny);
    if getLR
        LR_noise = noise * randn(nx,ny);
        LR = rOD(1) + (rOD(2)-rOD(1))*rand(nx,ny);
        %     LR = zeros(nx,ny);
    else
        LR = zeros(nx,ny);
    end
    e = exp(linspace(log(1),log(ecc+1),nx*resol))-1;
    band_e = exp(linspace(log(1),log(ecc+1),nod+1))-1;
    p = -pi/2;
    w = dipole(e,p,a,b,k);
    bx = real(w);
    by = imag(w);
    
    p = pi/2;
    w = dipole(e,p,a,b,k);
    tx = real(w);
    ty = imag(w);
    assert(sum(bx-tx)==0);
    
    e = ecc;
    p = linspace(-pi/2, pi/2,ny*resol);
    w = dipole(e,p,a,b,k);
    rx = real(w);
    ry = imag(w);
    
    tw0 = sum(sqrt(diff(tx).^2 + diff(ty).^2))/nod;
    rate0 = 1/(6*tw0);
    
    x0 = [bx rx(1:ny*resol/2)]-k*log(a/b);
    by = [by ry(1:ny*resol/2)];
    ty = [ty fliplr(ry((ny*resol/2+1):ny*resol))];
    
    W = dipole(ecc,0,a,b,k)-k*log(a/b);
    d = (1+2/nx)*W/nx;
    x = linspace(-W/nx, W+W/nx, nx);
    W = W+2*W/nx;
    H = d*ny;
    y = linspace(-H/2, H/2, ny);
    
    for ix = 1:nx
        if x(ix) < x0(1) || x(ix) > x0(end)
            Pi(ix,:) = 0;
        else
            jx = find(x(ix)-x0 < 0, 1, 'first');
            assert(~isempty(jx));
            r = (x(ix) - x0(jx-1))/(x0(jx)-x0(jx-1));
            yb = by(jx-1) + r*(by(jx)-by(jx-1));
            yt = ty(jx-1) + r*(ty(jx)-ty(jx-1));
            for iy = 1:ny
                if y(iy) < yb || y(iy) > yt
                    Pi(ix,iy) = 0;
                end
            end
        end
    end
    if plot_patch > 0
        if ishandle(plot_patch)
            close(plot_patch);
        end
        figure(plot_patch);
        daspect([1,1,1]);
        xlim([x0(1), x0(end)]);
        hold on
        plot(x0,ty,'k');
        plot(x0,by,'k');
    end
    if getLR
        i0 = randi([1,2]);
        lr = rOD(i0);
        current_x0 = 1;
        np = ny*resol;
        top_polar = fliplr(linspace(0,pi/2,np)); %closing in
        bot_polar = linspace(-pi/2,0,np);
        
        crit_ecc = 0.75;
        icrit = find(crit_ecc - band_e >= 0, 1, 'last');
        icrit = 10;
        nband = nod+1;
        for ie = 2:nband
            tw = tw0*1;
            pp = 0.5;
            %rate = rate0*(pp + (1+pp-pp)*rand*(nband-ie)/nband);
            if ie <= icrit
                %       bp = bot_polar;
                %       tp = top_polar;
                %   else
                polar_pick = ceil(np*(ie)/(nband+50));
                bp = bot_polar(1:polar_pick);
                tp = top_polar(1:polar_pick);
                %         end
                crit_bound = -0.25;
                rate = rate0*3*(icrit-ie)/icrit;
                na = [1,3];
                %   if ie > 3
                %       crit_bound = ie^(1.15)/5;
                %   end
            else
                polar_pick = ceil(np*(ie)/(nband+ie^1.5));
                %   polar_pick = ceil(np*(ie-icrit/2)/(nband+1.5*ie));
                bp = bot_polar(1:polar_pick);
                tp = top_polar(1:polar_pick);
                %   bp = bot_polar;
                %   tp = top_polar;
                crit_bound = -0.5;
                pp = 0.5;
                rate = rate0*(pp + (1-pp)*(nband-icrit-ie)/(nband-icrit));
                na = [2,3];
            end
            %         end
            iso_ecc = dipole(band_e(ie),bp,a,b,k);
            bot_rx = real(iso_ecc)-k*log(a/b);
            bot_ry = imag(iso_ecc);
            iso_ecc = dipole(band_e(ie),tp,a,b,k);
            top_rx = real(iso_ecc)-k*log(a/b);
            top_ry = imag(iso_ecc);
            %len_bot = sqrt(diff(bot_rx).^2 + diff(bot_ry).^2);
            %len_top = sqrt(diff(top_rx).^2 + diff(top_ry).^2);
            %        if ie <= icrit
            %             [bot_rx, bot_ry] = crude_flip(bot_rx, bot_ry, bot_rx(1), bot_ry(1));
            %             [top_rx, top_ry] = crude_flip(top_rx, top_ry, top_rx(1), top_ry(1));
            %         else
            %             if band_e(ie) > 1.75
            %         if ie == icrit
            %                 [bot_rx, bot_ry, kb] = end_flip(bot_rx, bot_ry, 'head');
            %                 [top_rx, top_ry, kt] = end_flip(top_rx, top_ry, 'head');
            %         else
            %             if ie > icrit
            %                 [bot_rx, bot_ry] = end_flip(bot_rx, bot_ry, 'head', kb);
            %                 [top_rx, top_ry] = end_flip(top_rx, top_ry, 'head', kt);
            %             else
            [bot_rx, bot_ry] = end_flip(bot_rx, bot_ry, 'head');
            [top_rx, top_ry] = end_flip(top_rx, top_ry, 'head');
            %             end
            %         end
            %         end
            if ie < icrit
                [xb_ext, yb_ext] = extend_line(bot_rx, bot_ry, 'tail',struct('dim','y','bound',-crit_bound));
                bot_rx = [bot_rx, xb_ext];
                bot_ry = [bot_ry, yb_ext];
                [xt_ext, yt_ext] = extend_line(top_rx, top_ry, 'tail',struct('dim','y','bound',crit_bound));
                top_rx = [top_rx, xt_ext];
                top_ry = [top_ry, yt_ext];
            else
                if ie == icrit
                    [xb_ext, yb_ext, kb] = extend_line(bot_rx, bot_ry, 'tail',struct('dim','y','bound',-crit_bound));
                    bot_rx = [bot_rx, xb_ext];
                    bot_ry = [bot_ry, yb_ext];
                    [xt_ext, yt_ext, kt] = extend_line(top_rx, top_ry, 'tail',struct('dim','y','bound',crit_bound));
                    top_rx = [top_rx, xt_ext];
                    top_ry = [top_ry, yt_ext];
                else
                    [xb_ext, yb_ext] = extend_line(bot_rx, bot_ry, 'tail',struct('dim','y','bound',-crit_bound,'k',kb));
                    bot_rx = [bot_rx, xb_ext];
                    bot_ry = [bot_ry, yb_ext];
                    [xt_ext, yt_ext] = extend_line(top_rx, top_ry, 'tail',struct('dim','y','bound',crit_bound,'k',kt));
                    top_rx = [top_rx, xt_ext];
                    top_ry = [top_ry, yt_ext];
                end
            end
            len_bot = [0, cumsum(sqrt(diff(bot_rx).^2 + diff(bot_ry).^2))];
            len_top = [0, cumsum(sqrt(diff(top_rx).^2 + diff(top_ry).^2))];
            if plot_patch > 0
                plot(bot_rx, bot_ry);
                plot(top_rx, top_ry);
            end
            if ie == 2
                bot_lx = linspace(x0(1),bot_rx(end),length(bot_rx));
                bot_ly = zeros(size(bot_lx));
                top_lx = linspace(x0(1),top_rx(end),length(top_rx));
                top_ly = zeros(size(top_lx));
                if plot_patch > 0
                    plot(bot_lx, bot_ly);
                    plot(top_lx, top_ly);
                end
            end
            if rand > 0.5
                LR = assignLR(Pi, x, y, bot_lx, bot_ly, bot_rx, bot_ry, current_x0, nx, ny, LR, lr, LR_noise, -1, plot_patch, tw, rate, len_bot, na);
                [LR, current_x0] = assignLR(Pi, x, y, top_lx, top_ly, top_rx, top_ry, current_x0, nx, ny, LR, lr, LR_noise, 1, plot_patch, tw, rate, len_top, na);
            else
                LR = assignLR(Pi, x, y, top_lx, top_ly, top_rx, top_ry, current_x0, nx, ny, LR, lr, LR_noise, 1, plot_patch, tw, rate, len_top, na);
                [LR, current_x0] = assignLR(Pi, x, y, bot_lx, bot_ly, bot_rx, bot_ry, current_x0, nx, ny, LR, lr, LR_noise, -1, plot_patch, tw, rate, len_bot, na);
            end
            lr = rOD(mod(i0+ie-2,2)+1);
            bot_lx = bot_rx;
            bot_ly = bot_ry;
            top_lx = top_rx;
            top_ly = top_ry;
        end
        %     LR(LR>rOD(2)) = rOD(2);
        %     LR(LR<rOD(1)) = rOD(1);
        if plot_patch > 0
            title(num2str(icrit));
        end
    end
end

function w = dipole(ecc, polar, a, b, k)
    w = k*log((ecc.*exp(1j*polar.*f(ecc,polar,a))+a)./(ecc.*exp(1j*polar.*f(ecc,polar,b))+b));
end

function anisotropy = f(ecc, polar, ab)
    s1 = 0.76;
    s2 = 0.1821;
    anisotropy = cosh(polar).^(-2*s2./((ecc/ab).^s1 + (ecc/ab).^(-s1)));
    anisotropy(isnan(anisotropy)) = 0;
end

function [LR, current_x0] = assignLR(Pi, x, y, lx, ly, rx, ry, current_x0, nx, ny, LR, lr, LR_noise, ht, plot_patch, tw, rate, len, na)
    band_started = false;
    lr = ht*lr;
    if ht > 0
        yrange = ny:-1:(ny/2+1);
    else
        yrange = 1:ny/2;
    end
    min_dis = 3*tw;
    turn = zeros(round(len(end)/min_dis),2)-1;
    rr = randn;
    if rr < 0.3
        rr = 0.3;
    end
    rate = (1 + rr)*rate;
    pos = log(rand)/rate;
    %pos = -rand/rate;
    %pos = 0;
    midp = zeros(size(turn,1),1);
    change = zeros(length(len),2);
    il = 0;
    while pos < len(end)
        pos = pos - log(rand)/rate;
%        pos = pos + 1/rate;
        if pos >= len(end)
            break;
        end
        if pos + tw > len(1)
            il = il+1;
            turn(il,1) = pos;
            turn(il,2) = randi(na);
            if il>1
                backstab = turn(il,2) == 2 && ht > 0 || turn(il,2) == 3 && ht < 0;
                forward = turn(il-1,2) == 3 && ht > 0 || turn(il-1,2) == 2 && ht < 0;
                if backstab && forward
                    turn(il,1) = turn(il,1) + tw*sqrt(2);
                    if turn(il,1) >= len(end)
                        turn(il,1) = -1;
                        turn(il,2) = -1;
                        il = il -1;
                        break;
                    end
                end
            end
            if turn(il,2) ~= 1
                pos = pos + min_dis*sqrt(2);
            else
                pos = pos + min_dis;
            end
        end
    end
    j0 = 1;
    %count = 0;
    for i = 1:il
        for j = j0:length(len)
            if len(j) > turn(i,1) && len(j) <= turn(i,1) + tw
                change(j,1) = turn(i,2);
                change(j,2) = i;
                if j == 1
                    midp(i) = j;
                    if plot_patch
                        plot(rx(j), ry(j), 'sk');
                    end
                else
                    if len(j-1) <= turn(i,1)
                        midp(i) = j;
                        if plot_patch
                            switch change(j,1)
                                case 1
                                    plot(rx(j), ry(j),'oy');
                                case 2
                                    plot(rx(j), ry(j),'og');
                                case 3
                                    plot(rx(j), ry(j),'ob');
                            end
                        end
                    end
                end
                j0 = j;
                %count = count + 1;
            end
        end
    end
    rrx = zeros(length(rx),il);
    rry = zeros(length(ry),il);
    theta = pi/4;
    f45 = [cos(-theta), -sin(-theta); sin(-theta), cos(-theta)];
    b45 = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    for i = 1:il
        vec = [rx - rx(midp(i)); ry - ry(midp(i))]; 
        if turn(i,2) == 2
            tmp = f45*vec;
        else
            if turn(i,2) == 3
                tmp = b45*vec;
            end
        end
        if turn(i,2) ~= 1
            rrx(:,i) = tmp(1,:)'+rx(midp(i));
            rry(:,i) = tmp(2,:)'+ry(midp(i));
        end
    end
    %assert(count == il);
    %figure(106)
    %turn(1:il)
    %plot(change);
    for ix = current_x0:nx
        %disp([num2str(ie),':',num2str(ix)]);
        if x(ix) < min(lx)
            continue;
        else
            if ~band_started
                band_started = true;
                current_x0 = ix;
            else
                if x(ix) > max(rx)
                    %disp('break!');
                    break;
                end
            end
        end
        for iy = yrange
            assert(y(iy)*ht>0);
            r2left = false;
            jl = find(abs(y(iy))-abs(ly) >= 0, 1, 'first');
            if ~isempty(jl)
                if jl > 1
                    r = (y(iy) - ly(jl-1))/(ly(jl)-ly(jl-1));
                    xl = lx(jl-1) + r*(lx(jl)-lx(jl-1));
                else
                    xl = lx(1);
                end
                if x(ix) >= xl && Pi(ix,iy) == 1
                    r2left = true;
                end
            end
            
            l2right = false;
            jr = find(abs(y(iy))-abs(ry) >= 0, 1, 'first');
            if ~isempty(jr)
                if jr > 1
                    r = (y(iy) - ry(jr-1))/(ry(jr)-ry(jr-1));
                    xr = rx(jr-1) + r*(rx(jr)-rx(jr-1));
                else
                    xr = rx(1);
                end
                if x(ix) < xr && Pi(ix,iy) == 1
                    l2right = true;
                end
            end
            if l2right && r2left
                %                 LR(ix,iy) = LR(ix,iy) + lr - sign(lr) * LR_noise(ix,iy);
                %disp(['jr:',num2str(jr),', nr:',num2str(length(rx)),'=',num2str(1-(2*jr/length(rx)-1)^2)]);
                val = lr - sign(lr) * LR_noise(ix,iy);% *(1.5-(2*jr/length(rx)-1)^2);
                % orth
                D2 = ((x(ix)-rx).^2 + (y(iy) - ry).^2);
                [~, j] = min(D2);   
                if change(j,1) == 1
                    LR(ix,iy) = -val;
                else
                    assigned = false;
                    for i = 1:il
                        if turn(i,2) > 1
                            D2 = ((x(ix)-rrx(:,i)).^2 + (y(iy) - rry(:,i)).^2);
                            [~, j] = min(D2);
                            if change(j,1) == turn(i,2) && change(j,2) == i
                                LR(ix,iy) = -val;
                                assigned = true;
                                break;
                            end
                        end
                    end
                    if ~assigned
                        LR(ix, iy) = val;
                    end
                end
                if plot_patch > 0
                    if val == LR(ix,iy) && lr > 0 || -val == LR(ix,iy) && lr < 0
                        plot(x(ix),y(iy), 'or', 'MarkerSize', 1.5);
                    else
                        plot(x(ix),y(iy), 'ok', 'MarkerSize', 1.5);
                    end
                end
            end
        end
    end
end
