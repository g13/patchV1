% [gx,gy,gt,gr] = myGrad(G,mu,B1_ind,Pi,[,whichper])
% Gradients for an elastic net.
%
% mu is a list of M reference vectors (each of dimension D) and G
% contains the width and length of the net (for cortical map modelling,
% this means there are D maps defined over a rectangular cortex).
% ENgrad returns the numerical gradient of each dimension (of each map)
% in Cartesian (gx, gy) and polar (gt, gr) coordinates.
% For those variables which are periodic, a correction is performed to
% avoid spurious discontinuities at the period ends.
%
% Notes:
%
% - The gradient is computed as a first-order forward difference for
%   boundary points and as a first-order centred difference for interior
%   points -- which is what Matlab's `gradient' function does.
%
% - For periodic variables (of range R) in a 2D net, the gradients should
%   not exceed R/2 per pixel (because of the wraparound), and indeed both gx
%   and gy are always less or equal than R/2. But gt = sqrt(gx²+gy²) can
%   occasionally be larger than R/2 along the boundaries of the
%   net. Specifically, the worst cases are:
%   . The four corners: both gx and gy can be at most R/2 and so gr can be
%     at most sqrt(2)*R/2, i.e., 41% too large.
%   . The four edges: gx can be at most R/2 and gy at most R/4 (or vice
%     versa) and so gr can be at most sqrt(5)*R/4, i.e., 12% too large.
%   Both situations can only occur if there is a large discontinuity right
%   at the corner or edge. I could e.g. reset gr to R/2 in that case, but
%   since it is vary rare I don't bother.
%
% - ENgrad also works for 1D nets (where G = [G1]).
%
% In:
%   G: list of grid lengths [G1 G2]. The elastic net dimensionality L is
%      the length of G; the number M of grid points (knots) is the product
%      of the grid lengths: M = G1*G2.
%   mu: MxD matrix of M D-dimensional row vectors containing the value
%      for the reference vectors.
%   whichper: ?x3 matrix where each row contains the index and range of a
%      reference vector variable which is periodic in the range given.
%      For example, if variables 1 and 3 are periodic in [-pi,pi] and
%      [1,2], respectively, then whichper = [1 -pi pi;3 1 2].
%      Default: [] (no periodic variables).
% Out:
%   gx,gy,gt,gr: G1xG2xD matrices of D-dimensional vectors containing the
%      gradients, respectively as follows: x component, y component
%      (Cartesian coordinates), angle theta and modulus r (polar
%      coordinates). Thus, gr(i,j,d) is the gradient modulus of dimension
%      d at location (i,j) in the rectangle.

% Copyright (c) 2002 by Miguel A. Carreira-Perpinan

function [gx,gy,gt,gr] = myGrad(G,mu,B1_ind,I1_ind,Pi,whichper,gtEstList,checkGrad,prefix)

L = length(G);                  	% Net dimensionality
if L == 1 G = [G 1]; end        	% So that both L=1 and L=2 work
[M,D] = size(mu);
[b_r, b_c] = ind2sub(G,B1_ind);
[i_r, i_c] = ind2sub(G,I1_ind);
% Argument defaults
if ~exist('Pi','var') || isempty(Pi)
    Pi = ones(1,M)/M;
end
if ~exist('whichper','var') whichper = []; end

F = reshape(mu,[G D]);          	% F is G1xG2xD
gx = zeros(size(F)); gy = gx;

switch L
    case 1
        for d=1:D
            tmpx = gradient(F(:,:,d));
            gx(:,:,d) = tmpx;
        end
        % Correct periodic variables
        gx = ENgradcorr(gx,whichper,2);
    case 2
        for d=1:D
            if d==1
                [tmpx,tmpy,fdx,cdx,fdy,cdy] = myGradient(F(:,:,d), [b_r, b_c], [i_r, i_c], Pi);
            else
                [tmpx,tmpy] = myGradient(F(:,:,d), [b_r, b_c], [i_r, i_c], Pi);
            end
            gx(:,:,d) = tmpx;
            gy(:,:,d) = tmpy;
        end
        % Correct periodic variables
        gx = myGradcorr(gx,fdx,cdx,whichper);
        gy = myGradcorr(gy,fdy,cdy,whichper);
        if checkGrad
            for d=1:D
                figure;
                quiver(1:G(1),1:G(2),gx(:,:,d)',gy(:,:,d)', 'LineWidth', 0.1);
				daspect([1,1,1]);
				print(gcf,'-loose','-r1200','-dpng', [prefix,'gradCheck-var', num2str(d)]);
				close(gcf);
            end
        end
end
[gt,gr] = cart2pol(gx,gy);
for i = gtEstList
	gt_b = gtEstimate(gt(:,:,i), gr(:,:,i), Pi);
	b_ind = sub2ind([G,D],b_r,b_c,i);
	gt(b_ind) = gt_b(B1_ind);
	[gx(b_ind), gy(b_ind)] = pol2cart(gt(b_ind), gr(b_ind));
	if checkGrad
    	figure;
    	quiver(1:G(1),1:G(2),gx(:,:,i)',gy(:,:,i)', 'LineWidth', 0.1);
		daspect([1,1,1]);
		print(gcf,'-loose','-r1200','-dpng', [prefix,'gradRebound-var', num2str(i)]);
		close(gcf);
	end
end
% Return as list of M reference vectors
gx = reshape(gx,M,D); gy = reshape(gy,M,D);
gt = reshape(gt,M,D); gr = reshape(gr,M,D);
% For periodic variables, check gr is less than range/2
for i=1:size(whichper,1)
    tmp = sum(gr(logical(Pi),whichper(i,1)) > diff(whichper(i,2:3)/2)+5*eps);
    if tmp > 0 
        disp(['Note: for periodic variable ' num2str(whichper(i,1)) ...
            ', ' num2str(tmp) ' of ' num2str(sum(logical(Pi))) ...
            ' gradient moduli exceed half the range']);
    end
end
end

function [gx, gy, fdx, cdx, fdy, cdy] = myGradient(v, bp, ip, Pi)
    G = size(v);
    gx = zeros(G);
    gy = gx;
	% extend for difference calculation, unnecessary when G has extended to contain the regime before hand
    Pi_ex = reshape(Pi,G);
    Pi_ex = [zeros(G(1),1), Pi_ex, zeros(G(1),1)];
    Pi_ex = [zeros(1,G(2)+2); Pi_ex; zeros(1,G(2)+2)];
    G_ex = [G(1)+2, G(2)+2];
    % boundary points, forward difference
	% row is x:1, column is y:2
    xr = Pi_ex(sub2ind(G_ex,1+bp(:,1)+1,1+bp(:,2)));
    xl = Pi_ex(sub2ind(G_ex,1+bp(:,1)-1,1+bp(:,2)));
    yu = Pi_ex(sub2ind(G_ex,1+bp(:,1),1+bp(:,2)+1));
    yd = Pi_ex(sub2ind(G_ex,1+bp(:,1),1+bp(:,2)-1));
    
    xpick = xr & xl;
    cd = sub2ind(G,bp(xpick,1),bp(xpick,2));
    gx(cd) = 0.5*(v(sub2ind(G,bp(xpick,1)+1,bp(xpick,2)))-v(sub2ind(G,bp(xpick,1)-1,bp(xpick,2))));
    if nargout > 2
        cdx = cd;
    end
    xpick = xr & ~xl;
    fd = sub2ind(G,bp(xpick,1),bp(xpick,2));
    gx(fd) = v(sub2ind(G,bp(xpick,1)+1,bp(xpick,2)))-v(sub2ind(G,bp(xpick,1),bp(xpick,2)));
    if nargout > 2
        fdx = fd;
    end
    
    xpick = ~xr & xl;
    fd = sub2ind(G,bp(xpick,1),bp(xpick,2));
    gx(fd) = v(sub2ind(G,bp(xpick,1),bp(xpick,2)))-v(sub2ind(G,bp(xpick,1)-1,bp(xpick,2)));
    if nargout > 2
        fdx = [fdx; fd];
    end
    
    ypick = yu & yd;
    cd = sub2ind(G,bp(ypick,1),bp(ypick,2));
    gy(cd) = 0.5*(v(sub2ind(G,bp(ypick,1),bp(ypick,2)+1))-v(sub2ind(G,bp(ypick,1),bp(ypick,2)-1)));
    if nargout > 2
        cdy = cd;
    end
    
    ypick = yu & ~yd;
    fd = sub2ind(G,bp(ypick,1),bp(ypick,2));
    gy(fd) = v(sub2ind(G,bp(ypick,1),bp(ypick,2)+1))-v(sub2ind(G,bp(ypick,1),bp(ypick,2)));
    if nargout > 2
        fdy = fd;
    end

    ypick = ~yu & yd;
    fd = sub2ind(G,bp(ypick,1),bp(ypick,2));
    gy(fd) = v(sub2ind(G,bp(ypick,1),bp(ypick,2)))-v(sub2ind(G,bp(ypick,1),bp(ypick,2)-1));
    if nargout > 2
        fdy = [fdy; fd];
    end
    
    % interior points center difference
    cd = sub2ind(G,ip(:,1),ip(:,2));
    gx(cd) = 0.5*(v(sub2ind(G,ip(:,1)+1,ip(:,2)))-v(sub2ind(G,ip(:,1)-1,ip(:,2))));
    gy(cd) = 0.5*(v(sub2ind(G,ip(:,1),ip(:,2)+1))-v(sub2ind(G,ip(:,1),ip(:,2)-1)));
    if nargout > 2
        cdx = [cdx; cd];
        cdy = [cdy; cd];
    end
end

function Mc = myGradcorr(M,fdI,cdI,whichper)
    Mc = reshape(M, size(M,1)*size(M,2),size(M,3));
    for i=1:size(whichper,1)
        v = whichper(i,1); range = diff(whichper(i,2:3));
        % Interior points (central difference)
        tmpi = Mc(cdI,v);
        f = find(abs(tmpi) > range/4); 
        tmpi(f) = tmpi(f) - sign(tmpi(f))*range/2;
        Mc(cdI,v) = tmpi;
        % Boundary points (forward or backward difference)
        tmpb = Mc(fdI,v);
        f = find(abs(tmpb) > range/2); 
        tmpb(f) = tmpb(f) - sign(tmpb(f))*range;
        Mc(fdI,v) = tmpb;
    end
    Mc = reshape(Mc, size(M));
end

% Mc = ENgradcorr(M,whichper,d)
%
% For periodic variables, correct fake discontinuities at the period ends.
%
% In:
%   M: ?x?xD matrix of D-dimensional vectors representing gradient components
%      along the dimension d.
%   whichper: ?x3 matrix as in ENgrad.
%   d: dimension to which the gradient components correspond: 1 for x, 2 for y.
% Out:
%   Mc: corrected M matrix.

function Mc = ENgradcorr(M,whichper,d)

% Idea: a given variable that depends on the grid location in a continuous
% but discretised way will show small jumps between consecutive values. But
% if the variable is periodic, there will be large jumps when the values
% cross one end of the period interval and come out at the other end.
% In the gradient of the variable, these show up as large discontinuities.
% We can avoid such spurious discontinuities by finding all such large
% jumps and correcting them.
%
% The localisation and correction must be done separately for the boundary
% points (where Matlab has applied a forward or backward difference) and
% the interior points (where Matlab has applied a central difference):
%                              ---is this right?---
% - If using the forward difference (v1-v0)/2, a jump larger than half the
%   range (in absolute value) must be corrected (by adding or subtracting
%   the range), since the jump is shorter going in the other direction. That
%   is, all jumps must be at most equal to half the range.
%
% - If using the central difference (v2-v0)/2, the rule is "a jump larger
%   than 1/4 of the range (in absolute value) must b corrected (by adding or
%   subtracting half the range)". That is, all jumps must be at most equal
%   to 1/4 of the range.
%
% See note at the top regarding too large a gradient modulus gr.

Mc = M;
for i=1:size(whichper,1)
    v = whichper(i,1); range = diff(whichper(i,2:3));
    % Interior points (central difference)
    if d==2
        tmpi = Mc(2:end-1,:,v);
    else
        tmpi = Mc(:,2:end-1,v);
    end
    f = find(abs(tmpi) > range/4); 
    tmpi(f) = tmpi(f) - sign(tmpi(f))*range/2;
    if d==2
        Mc(2:end-1,:,v) = tmpi;
    else
        Mc(:,2:end-1,v) = tmpi;
    end
    % Boundary points (forward or backward difference)
    if d==2
        tmpb = Mc([1 end],:,v);
    else
        tmpb = Mc(:,[1 end],v);
    end
    f = find(abs(tmpb) > range/2); 
    tmpb(f) = tmpb(f) - sign(tmpb(f))*range;
    if d==2
        Mc([1 end],:,v) = tmpb;
    else
        Mc(:,[1 end],v) = tmpb;
    end
end
end
% Note: maybe this can be simplified using Matlab's `unwrap' function.
