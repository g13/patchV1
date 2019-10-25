% VF is in the form of (n,2) each entry is (eccentricity, polar)
% resol is number of data points to be interpolated.
function VF = cortex2VF(xy, ecc, resol, a, b, k)
    if nargin < 6
        k = sqrt(140)*0.873145;
        if nargin < 5
            b = 96.7; 
            if nargin < 4
                a = 0.635; 
                if nargin < 3
                    resol = 100;
                end
            end
        end
    end
    n = size(xy,1);
    VF = zeros(n,2);
	me = min(max(n*resol,100),10000);
	mp = min(max(n*resol,100),10000);
	log_e = linspace(log(1), log(ecc+1), me);
    e = exp(log_e)-1;
    p = linspace(-pi/2, pi/2, mp);
	vx = zeros(me, mp);
	vy = zeros(me, mp);
	for ip = 1:mp
		w = dipole(e,p(ip),a,b,k)-k*log(a/b);
		vx(:,ip) = real(w);
		vy(:,ip) = imag(w);
	end
	ix0 = zeros(mp-1, 1);
	d0 = zeros(mp-1, 1);
	d1 = zeros(mp-1, 1);
	prange = 1:mp-1;
	for i = 1:n
		mask = (xy(i,1) - vx(1:me-1,1:mp-1) >= 0 & xy(i,1) - vx(2:me,1:mp-1) <= 0);
		pmask = any(mask,1);
		if ~any(pmask)
			print('point not in bound');
            assert(0);
		end
		assert(length(pmask) == mp-1);
		pnull = ~pmask;
		[ix, ~] = find(mask);
		assert(sum(pmask) == length(ix));
		ix0(pmask) = ix;
		ix0(pnull) = -1;
	    ind = sub2ind([me,mp],ix,prange(pmask)');
	    d0(pmask) = (vx(ind) - xy(i,1)).^2 + (vy(ind) - xy(i,2)).^2;
	    ind = sub2ind([me,mp],ix+1,prange(pmask)');
	    d1(pmask) = (vx(ind) - xy(i,1)).^2 + (vy(ind) - xy(i,2)).^2;
	    d0(pnull) = inf;
	    d1(pnull) = inf;
	    dis = min([d0, d1], [], 2);
	    assert(length(dis) == mp-1);
	    [~, idp] = min(dis);
	    idx = ix0(idp);
	    %VF(i,j,1) =  log_e(idx) + (log_e(idx+1) - log_e(idx)) * sqrt(dis(idp))/(sqrt(d0(idp))+sqrt(d1(idp)));
	    VF(i,1) =  exp(log_e(idx) + (log_e(idx+1) - log_e(idx)) * sqrt(dis(idp))/(sqrt(d0(idp))+sqrt(d1(idp))))-1;
	    %w = dipole(exp(VF(i,j,1))-1,p(idp),a,b,k) - k*log(a/b);
	    w = dipole(VF(i,1),p(idp),a,b,k) - k*log(a/b);
        vp_x0 = real(w);
        vp_y0 = imag(w);
	    %w = dipole(exp(VF(i,j,1))-1,p(idp+1),a,b,k) - k*log(a/b);
	    w = dipole(VF(i,1),p(idp+1),a,b,k) - k*log(a/b);
        vp_x1 = real(w);
        vp_y1 = imag(w);
        dp0 = sqrt((xy(i,1)-vp_x0)^2 + (xy(i,2)-vp_y0)^2);
        dp1 = sqrt((xy(i,1)-vp_x1)^2 + (xy(i,2)-vp_y1)^2);
        VF(i,2) = p(idp) + (p(idp+1) - p(idp)) * dp0/(dp0+dp1);
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
