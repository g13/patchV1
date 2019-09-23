function [Pi, W, LR, VF, VFy_ratio] = myCortex(stream, G, xrange, tr_x, yrange, tr_y, VFweights, ecc, a, b, k, resol, nod, rOD, noise, manual_LR, plot_patch,savepath,scale_VFy)
    nx = G(1);
    ny = G(2);
    Pi = ones(nx,ny);
    VF = zeros(nx,ny,2);
    VFy_ratio = zeros(nx,ny);
    if manual_LR
        LR_noise = noise * randn(stream,nx,ny);
        LR = rOD(1) + (rOD(2)-rOD(1))*rand(stream,nx,ny);
        %     LR = zeros(nx,ny);
    else
        LR = zeros(nx,ny);
    end
	log_e = linspace(log(1),log(ecc+1),nx*resol);
    e = exp(log_e)-1;
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
    
    p = linspace(-pi/2, pi/2,ny*resol);
	w = dipole(e(end),p,a,b,k);
    rx = real(w);
    ry = imag(w);
    max_y = max(ry);
    
    tw0 = sum(sqrt(diff(tx).^2 + diff(ty).^2))/nod;
    rate0 = 1/(6*tw0);
    
    x0 = [bx rx(1:ny*resol/2)]-k*log(a/b);
    by = [by ry(1:ny*resol/2)];
    ty = [ty fliplr(ry((ny*resol/2+1):ny*resol))];
    
    W = dipole(ecc,0,a,b,k)-k*log(a/b);
    x = linspace(-W/(2*nx-4), W+W/(2*nx-4), nx);
	disp(['starting at x = ', num2str((x(1) + x(2))/2)]);
    W = W+W/(nx-2);
	d = W/(nx-1); 
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
        figure;
		subplot(2,2,1)
        daspect([1,1,1]);
        xlim([x0(1), x0(end)]);
        hold on
        plot(x0,ty,'k');
        plot(x0,by,'k');
		xlabel('Initial mu interpolation points density x 0.01');
    end
	% ecc-polar grid for VF see COMMENTs in function assign_pos_VF of assign_attr.py 
	me = nx*resol;
	mp = ny*resol;
	vx = zeros(me, mp);
	vy = zeros(me, mp);
	for ip = 1:mp
		w = dipole(e,p(ip),a,b,k)-k*log(a/b);
		vx(:,ip) = real(w);
		vy(:,ip) = imag(w);
        if plot_patch >0 && mod(ip,resol*10) == 0
			subplot(2,2,1)
            plot(vx(:,ip), vy(:,ip),'k');
        end
	end
    if plot_patch >0
		subplot(2,2,1)
	    for ie = 1:me
            if mod(ie,resol*10) == 0
                plot(vx(ie,:), vy(ie,:),'r');
            end
        end
		subplot(2,2,3)
        hold on
        pcolor(tr_x(1:end-1,1:end-1), tr_y(1:end-1,1:end-1),log(VFweights));
		pcolor(tr_x(1:end-1,1:end-1),-tr_y(1:end-1,1:end-1),log(VFweights));
        % plot outbounds
        plot(tr_x(end,:),tr_y(end,:),'-r');
        plot(tr_x(end,:),-tr_y(end,:),'-r');
        plot(tr_x(end-1,:),tr_y(end-1,:),'-r');
        plot(tr_x(end-1,:),-tr_y(end-1,:),'-r');
        plot([tr_x(:,end), flipud(tr_x(:,end))],[tr_y(:,end),-flipud(tr_y(:,end))],'-r');
        plot([tr_x(:,end-1), flipud(tr_x(:,end-1))],[tr_y(:,end-1),-flipud(tr_y(:,end-1))],'-r');
        for i=1:size(tr_x,2)
            plot(tr_x(end-1:end,:), tr_y(end-1:end,:),'-r');
            plot(tr_x(end-1:end,:),-tr_y(end-1:end,:),'-r');
        end
        for i=1:size(tr_x,1)
            plot(tr_x(:,end-1:end)', tr_y(:,end-1:end)','-r');
            plot(tr_x(:,end-1:end)',-tr_y(:,end-1:end)','-r');
        end
            
        plot(x0,ty,'-r');
        plot(x0,by,'-r');
        xlim([min([min(x0),min(tr_x(:))]),max([max(x0),max(tr_x(:))])]);
        ylim([min([min(by),min(tr_y(:))]),max([max(ty),max(tr_y(:))])]);
        daspect([1,1,1]);
		colormap(viridis);
		colorbar;
		xlabel('Training Points weights in log(area)');
    end
	ix0 = zeros(mp-1, 1);
	d0 = zeros(mp-1, 1);
	d1 = zeros(mp-1, 1);
	prange = 1:mp-1;
	f = 0;
	for i = 1:nx
		mask = (x(i) - vx(1:me-1,1:mp-1) > 0 & x(i) - vx(2:me,1:mp-1) < 0);
		pmask = any(mask,1);
		if ~any(pmask)
			continue;
		else
			f = f + 1;
		end
		assert(length(pmask) == mp-1);
		pnull = ~pmask;
		[ix, ~] = find(mask);
		assert(sum(pmask) == length(ix));
		ix0(pmask) = ix;
		ix0(pnull) = -1;
		for j = 1:ny
			if Pi(i,j) == 0
				continue;
			end
			ind = sub2ind([me,mp],ix,prange(pmask)');
			d0(pmask) = (vx(ind) - x(i)).^2 + (vy(ind) - y(j)).^2;
			ind = sub2ind([me,mp],ix+1,prange(pmask)');
			d1(pmask) = (vx(ind) - x(i)).^2 + (vy(ind) - y(j)).^2;
			d0(pnull) = inf;
			d1(pnull) = inf;
			dis = min([d0, d1], [], 2);
			assert(length(dis) == mp-1);
			[~, idp] = min(dis);
			idx = ix0(idp);
			%VF(i,j,1) =  log_e(idx) + (log_e(idx+1) - log_e(idx)) * sqrt(dis(idp))/(sqrt(d0(idp))+sqrt(d1(idp)));
			VF(i,j,1) =  exp(log_e(idx) + (log_e(idx+1) - log_e(idx)) * sqrt(dis(idp))/(sqrt(d0(idp))+sqrt(d1(idp))))-1;
			%w = dipole(exp(VF(i,j,1))-1,p(idp),a,b,k) - k*log(a/b);
			w = dipole(VF(i,j,1),p(idp),a,b,k) - k*log(a/b);
            vp_x0 = real(w);
            vp_y0 = imag(w);
			%w = dipole(exp(VF(i,j,1))-1,p(idp+1),a,b,k) - k*log(a/b);
			w = dipole(VF(i,j,1),p(idp+1),a,b,k) - k*log(a/b);
            vp_x1 = real(w);
            vp_y1 = imag(w);
            dp0 = sqrt((x(i)-vp_x0)^2 + (y(j)-vp_y0)^2);
            dp1 = sqrt((x(i)-vp_x1)^2 + (y(j)-vp_y1)^2);
            VF(i,j,2) = p(idp) + (p(idp+1) - p(idp)) * dp0/(dp0+dp1);
            %w = dipole(exp(VF(i,j,1))-1,pi/2,a,b,k);
            w = dipole(VF(i,j,1),pi/2,a,b,k);
            vp_y = imag(w);
            VFy_ratio(i,j) = vp_y/max_y;
            assert(VFy_ratio(i,j) <= 1.0);
		end
	end
    if plot_patch > 0
		if scale_VFy
			subplot(3,2,2)
		else
			subplot(2,2,2)
		end
		imagesc(VF(:,:,1));
		colormap(viridis);
		colorbar;
        daspect([1,1,1]);
		if scale_VFy
			subplot(3,2,4)
		else
			subplot(2,2,4)
		end
		imagesc(VF(:,:,2));
		colormap(viridis);
		colorbar;
        daspect([1,1,1]);
		if scale_VFy
			subplot(3,2,6)
			imagesc(VF(:,:,2).*VFy_ratio);
			colormap(viridis);
			colorbar;
        	daspect([1,1,1]);
		end
		if manual_LR
			subplot(1,2,1)
		end
	end
	pPi = reshape(Pi > 0, prod(G),1);
	VF = reshape(VF, prod(G), 2);
	maxVFx = max(max(VF(pPi,1)));
	minVFx = min(min(VF(pPi,1)));
	maxVFy = max(max(VF(pPi,2)));
	minVFy = min(min(VF(pPi,2)));
	disp('ecc range:');
	disp([minVFx, maxVFx]);
	disp('polar range:');
	disp([minVFy, maxVFy]);
	[VF(:,1), VF(:,2)] = pol2cart(VF(:,2), VF(:,1));

	maxVFx = max(max(VF(pPi,1)));
	minVFx = min(min(VF(pPi,1)));
	maxVFy = max(max(VF(pPi,2)));
	minVFy = min(min(VF(pPi,2)));
	disp('x range:');
	disp([minVFx, maxVFx]);
	disp('y range:');
	disp([minVFy, maxVFy]);
	VF(pPi,1) = xrange(1) + (VF(pPi,1) - minVFx)/(maxVFx - minVFx)*(xrange(2)-xrange(1));
	VF(pPi,2) = yrange(1) + (VF(pPi,2) - minVFy)/(maxVFy - minVFy)*(yrange(2)-yrange(1));
	VF(~pPi,:) = 0.0;
	VF = reshape(VF, [G, 2]);
    if manual_LR
        i0 = randi(stream,[1,2]);
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
            %rate = rate0*(pp + (1+pp-pp)*rand(stream)*(nband-ie)/nband);
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
            if rand(stream) > 0.5
                LR = assignLR(Pi, x, y, bot_lx, bot_ly, bot_rx, bot_ry, current_x0, nx, ny, LR, lr, LR_noise, -1, plot_patch, tw, rate, len_bot, na, stream);
                [LR, current_x0] = assignLR(Pi, x, y, top_lx, top_ly, top_rx, top_ry, current_x0, nx, ny, LR, lr, LR_noise, 1, plot_patch, tw, rate, len_top, na, stream);
            else
                LR = assignLR(Pi, x, y, top_lx, top_ly, top_rx, top_ry, current_x0, nx, ny, LR, lr, LR_noise, 1, plot_patch, tw, rate, len_top, na, stream);
                [LR, current_x0] = assignLR(Pi, x, y, bot_lx, bot_ly, bot_rx, bot_ry, current_x0, nx, ny, LR, lr, LR_noise, -1, plot_patch, tw, rate, len_bot, na, stream);
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
    if plot_patch > 0
		print(gcf,'-loose', '-r600', '-dpng', [savepath,'/cortical_OD_VF-',num2str(plot_patch),'.png']);
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

function [LR, current_x0] = assignLR(Pi, x, y, lx, ly, rx, ry, current_x0, nx, ny, LR, lr, LR_noise, ht, plot_patch, tw, rate, len, na, stream)
    band_started = false;
    lr = ht*lr;
    if ht > 0
        yrange = ny:-1:(ny/2+1);
    else
        yrange = 1:ny/2;
    end
    min_dis = 3*tw;
    turn = zeros(round(len(end)/min_dis),2)-1;
    rr = randn(stream);
    if rr < 0.3
        rr = 0.3;
    end
    rate = (1 + rr)*rate;
    pos = log(rand(stream))/rate;
    %pos = -rand(stream)/rate;
    %pos = 0;
    midp = zeros(size(turn,1),1);
    change = zeros(length(len),2);
    il = 0;
    while pos < len(end)
        pos = pos - log(rand(stream))/rate;
%        pos = pos + 1/rate;
        if pos >= len(end)
            break;
        end
        if pos + tw > len(1)
            il = il+1;
            turn(il,1) = pos;
            turn(il,2) = randi(stream,na);
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
