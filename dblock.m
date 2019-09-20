% reference patch_geo_func.py
function darea = dblock(e,p,k,a,b,s1,s2)
	if nargin < 7
		s2 = 0.1821;
		if nargin < 6
			s1 = 0.76;
		end
	end
    eval_fp_a = fp(e,p,a,s1,s2);
    d_eval_fpp_a_e = d_fpp_e(e,p,a,s1,s2,eval_fp_a);
    d_eval_fpp_a_p = d_fpp_p(e,p,a,s1,s2,eval_fp_a);
    cos_eval_fpp_a = cos(eval_fp_a.*p);
    sin_eval_fpp_a = sin(eval_fp_a.*p);
    R_ep_a = R_ep(e,p,a,s1,s2,eval_fp_a,cos_eval_fpp_a);

    eval_fp_b = fp(e,p,b,s1,s2);
    d_eval_fpp_b_e = d_fpp_e(e,p,b,s1,s2,eval_fp_b);
    d_eval_fpp_b_p = d_fpp_p(e,p,b,s1,s2,eval_fp_b);
    cos_eval_fpp_b = cos(eval_fp_b.*p);
    sin_eval_fpp_b = sin(eval_fp_b.*p);
    R_ep_b = R_ep(e,p,b,s1,s2,eval_fp_b,cos_eval_fpp_b);

    val = (a*cos_eval_fpp_a + e - a*e.*sin_eval_fpp_a.*d_eval_fpp_a_e)./R_ep_a;
    val = val - (b*cos_eval_fpp_b + e - b*e.*sin_eval_fpp_b.*d_eval_fpp_b_e)./R_ep_b;
    dxe = val*k;

    d_phi_a_p = e.*d_eval_fpp_a_p.*(e+a*cos_eval_fpp_a)./R_ep_a;
    d_phi_b_p = e.*d_eval_fpp_b_p.*(e+b*cos_eval_fpp_b)./R_ep_b;
    dyp = k*(d_phi_a_p - d_phi_b_p);

    val = -a*e.*sin_eval_fpp_a.*d_eval_fpp_a_p./R_ep_a;
    val = val + b*e.*sin_eval_fpp_b.*d_eval_fpp_b_p./R_ep_b;
    dxp = val*k;

    d_phi_a_e = (a*sin_eval_fpp_a + (a*e.*cos_eval_fpp_a + e.*e).*d_eval_fpp_a_e)./R_ep_a;
    d_phi_b_e = (b*sin_eval_fpp_b + (b*e.*cos_eval_fpp_b + e.*e).*d_eval_fpp_b_e)./R_ep_b;
    dye = k*(d_phi_a_e - d_phi_b_e);
	darea = dxe.*dyp - dxp.*dye;
end

function f = fp(e,p,ab,s1,s2)
	if nargin < 5
		s2 = 0.1821;
		if nargin < 4
			s1 = 0.76;
		end
	end
    if e == 0
        f = 0;
    else
        f = (1./cosh(p)).^(2*s2./((e/ab).^s1 + (e/ab).^(-s1)));
	end
end

function f = R_ep(e,p,ab,s1,s2,eval_fp,cos_eval_fp)
	if nargin < 7
		cos_eval_fp = nan;
		if nargin < 6
			eval_fp = nan;
			if nargin < 5
				s2 = 0.1821;
				if nargin < 4
					s1 = 0.76;
				end
			end
		end
	end

    if isnan(eval_fp)
        eval_fp = fp(e,p,ab,s1,s2);
	end
    if isnan(cos_eval_fp)
        cos_eval_fp = cos(eval_fp.*p);
	end
    f = ab.*ab + e.*e + 2*ab.*e.*cos_eval_fp;
end

function f = d_sech_log(e,ab,s1,s2)
    if e == 0
        f = 0;
    else
        e_ab_s1 = (e/ab).^s1;
        e_ab_s1r =(ab./e).^s1;
        f = -2*s2*s1./e.*(e_ab_s1-e_ab_s1r)./(e_ab_s1+e_ab_s1r).^2;
	end
end

function f = d_fp_e(e,p,ab,s1,s2,eval_fp)
	if nargin < 6
		eval_fp = nan;
		if nargin < 5
			s2 = 0.1821;
			if nargin < 4
				s1 = 0.76;
			end
		end
	end
	
    if e == 0
        f = 0;
    else
        if isnan(eval_fp)
			eval_fp = fp(e,p,ab,s1,s2);
		end
        f = eval_fp .* log(1./cosh(p)) .* d_sech_log(e,ab,s1,s2);
	end
end
    
function f = d_fp_p(e,p,ab,s1,s2,eval_fp)
	if nargin < 6
		eval_fp = nan;
		if nargin < 5
			s2 = 0.1821;
			if nargin < 4
				s1 = 0.76;
			end
		end
	end
		
	d_sech = @(x) -2*(exp(x)-exp(-x))./(exp(x)+exp(-x)).^2;

    if e == 0
        f = 0;
	else
    	if isnan(eval_fp)
        	eval_fp = fp(e,p,ab,s1,s2);
		end
        f = 2*s2./((e/ab).^s1 + (e/ab).^(-s1)) .* eval_fp .* cosh(p) .* d_sech(p);
	end
end
    
function f = d_fpp_e(e,p,ab,s1,s2,eval_fp)
	if nargin < 6
		eval_fp = nan;
		if nargin < 5
			s2 = 0.1821;
			if nargin < 4
				s1 = 0.76;
			end
		end
	end
	%assert(length(p) == length(e) || length(p) == 1 || length(e) == 1);
	%if length(e) > length(p)
	%	f = zeros(size(e));
	%	if length(p) == 1
	%		p = ones(size(e))*p;
	%	end
	%else
	%	f = zeros(size(p));
	%	if length(e) == 1
	%		e = ones(size(p))*e;
	%	end
	%end
	%predicate = p ~= 0;
    %f(predicate) = p(predicate).*d_fp_e(e(predicate),p(predicate),ab,s1,s2,eval_fp)
    f = p.*d_fp_e(e,p,ab,s1,s2,eval_fp);
end
    
function f = d_fpp_p(e,p,ab,s1,s2,eval_fp)
	if nargin < 6
		eval_fp = nan;
		if nargin < 5
			s2 = 0.1821;
			if nargin < 4
				s1 = 0.76;
			end
		end
	end
    if p == 0
        f = 1;
    else
        if isnan(eval_fp)
			eval_fp = fp(e,p,ab,s1,s2);
	   	end 
        f = p.*d_fp_p(e,p,ab,s1,s2,eval_fp) + eval_fp;
	end
end
