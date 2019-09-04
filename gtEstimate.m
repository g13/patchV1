% estimate weighted gt (in myV1stats.m) when single pixel gradient orientation is not sufficient
function gt = gtEstimate(gt0, gr, pick, Pi, h, centeroff)	
	if nargin < 5
		centeroff = true
		if nargin < 4
			h = 3;
		end
	end
	G = size(gt0);
	% padding
	gt_ex = [zeros(h,G(2)); gt0; zeros(h,G(2))];
	gt_ex = [zeros(G(1)+2*h,h), gt_ex, zeros(G(1)+2*h,h)];
	assert(size(gt_ex,1) == G(1) + 2*h);
	assert(size(gt_ex,2) == G(2) + 2*h);
	gr_ex = [zeros(h,G(2)); gr; zeros(h,G(2))];
	gr_ex = [zeros(G(1)+2*h,h), gr_ex, zeros(G(1)+2*h,h)];
	if centeroff
		gt = zeros(G);
		w = zeros(G);
	else
		gt = gt0.*gr;
		w = gr;
	end
	for d = 1:h
		% add elements on the left
		gt = gt + near(gt_ex(h+(1:G(1))-d, h+(1:G(2))), gt0) .* gr_ex(h+(1:G(1))-d,h+(1:G(2)));
		% add elements on the right
		gt = gt + near(gt_ex(h+(1:G(1))+d, h+(1:G(2))), gt0) .* gr_ex(h+(1:G(1))+d,h+(1:G(2)));
		% add elements on the top
		gt = gt + near(gt_ex(h+(1:G(1)), h+(1:G(2))+d), gt0) .* gr_ex(h+(1:G(1)),h+(1:G(2))+d);
		% add elements on the down
		gt = gt + near(gt_ex(h+(1:G(1)), h+(1:G(2))-d), gt0) .* gr_ex(h+(1:G(1)),h+(1:G(2))-d);
		% sum the weigthts

		% add elements on the bottom-left 
		gt = gt + near(gt_ex(h+(1:G(1))-d, h+(1:G(2))-d), gt0) .* gr_ex(h+(1:G(1))-d,h+(1:G(2))-d);
		% add elements on the bottom-right 
		gt = gt + near(gt_ex(h+(1:G(1))+d, h+(1:G(2))-d), gt0) .* gr_ex(h+(1:G(1))+d,h+(1:G(2))-d);
		% add elements on the top-left 
		gt = gt + near(gt_ex(h+(1:G(1))-d, h+(1:G(2))+d), gt0) .* gr_ex(h+(1:G(1))-d,h+(1:G(2))+d);
		% add elements on the top-right 
		gt = gt + near(gt_ex(h+(1:G(1))+d, h+(1:G(2))+d), gt0) .* gr_ex(h+(1:G(1))+d,h+(1:G(2))+d);

		% sum the weigthts
		w = w + gr_ex(h+(1:G(1))-d,h+(1:G(2)));
		w = w + gr_ex(h+(1:G(1))+d,h+(1:G(2)));
		w = w + gr_ex(h+(1:G(1)),h+(1:G(2))-d);
		w = w + gr_ex(h+(1:G(1)),h+(1:G(2))+d);
		w = w + gr_ex(h+(1:G(1))-d,h+(1:G(2))-d);
		w = w + gr_ex(h+(1:G(1))+d,h+(1:G(2))-d);
		w = w + gr_ex(h+(1:G(1))-d,h+(1:G(2))+d);
		w = w + gr_ex(h+(1:G(1))+d,h+(1:G(2))+d);
	end
	gt = gt./w;
	gt(gt>pi) = gt(gt>pi) - 2*pi;
	gt(gt<-pi) = gt(gt<-pi) + 2*pi;
end

function gt = near(gt, gt0)
	dg = gt - gt0;
	pick = abs(dg) > pi;
	gt(pick) = gt0(pick) - sign(dg(pick)) .* (2*pi-abs(dg(pick)));
end
