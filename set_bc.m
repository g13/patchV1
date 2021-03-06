function [B_ind, I_ind, B1_ind, I1_ind, B2_ind, I2_ind, B1_ang] = set_bc(Pi, bw, right_open, check_boundary, prefix, non_oblique)
	% assuming vertical symmetry
	% the open boundary is to the right
    [nx, ny] = size(Pi);
	if non_oblique
		right_most = nx;
	else
		right_most = 0;
	end
	
    B_ind = zeros(nx*ny,1); % boundary with width=bw, open if right_open
    B1_ind = zeros(nx*ny,1); % boundary with width=1, open if right_open
	if right_open
		%  close boundary with width=1
    	B2_ind = zeros(nx*ny,1);
		k = 0;
	end
    i = 0;
    j = 0;
	right_most_set = false;
    for iy = 1:ny
        head = find(Pi(:,iy),bw,'first'); 
        tail = find(Pi(:,iy),bw,'last');
        if ~isempty(head)
			if ~right_most_set && right_most < tail(end)
				right_most = tail(end);
				right_most_set = true;
			end
			if ~right_open
				inds = unique([head; tail]);
				ninds = length(inds);
            	B_ind(i+1:i+ninds) = (iy-1)*nx + inds;
            	i = i + ninds;
				if head(1) == tail(end)
            		B1_ind(j+1) = (iy-1)*nx + head(1);
            		j = j + 1;
				else 
					assert(head(1) < tail(end));
            		B1_ind(j+1:j+2) = (iy-1)*nx + [head(1); tail(end)];
            		j = j + 2;
				end
			else
				%not including the open boundary
            	B_ind(i+1:i+length(head)) = (iy-1)*nx + head;
            	i = i + length(head);
            	B1_ind(j+1) = (iy-1)*nx + head(1); 
            	j = j + 1;
				%including the open boundary
				if head(1) == tail(end)
            		B2_ind(k+1) = (iy-1)*nx + head(1);
            		k = k + 1;
				else
            		B2_ind(k+1:k+2) = (iy-1)*nx + [head(1); tail(end)];
            		k = k + 2;
				end
			end
        else
            assert(isempty(tail));
        end
    end
	assert(right_most >= 1);
    for ix = 1:right_most
        head = find(Pi(ix,:),bw,'first');
        tail = find(Pi(ix,:),bw,'last');
        if ~isempty(head)
			inds = unique([head; tail]);
			ninds = length(inds);
            B_ind(i+1:i+ninds) = (inds-1)*nx + ix;
            i = i + ninds;
			if head(1) == tail(end)
            	B1_ind(j+1) = (head(1)-1)*nx + ix;
            	j = j + 1;
				if right_open
            		B2_ind(k+1) = (head(1)-1)*nx + ix;
            		k = k + 1;
				end
			else
            	B1_ind(j+1:j+2) = ([head(1); tail(end)]-1)*nx + ix;
            	j = j + 2;
				if right_open
            		B2_ind(k+1:k+2) = ([head(1); tail(end)]-1)*nx + ix;
            		k = k + 2;
				end
			end
        else
            assert(isempty(tail));
        end
    end
    B_ind = B_ind(1:i);
    B1_ind = B1_ind(1:j);
	if ~right_open
		B2_ind = B1_ind;
	else
    	B2_ind = B2_ind(1:k);
	end
    Pi_tmp = ~Pi;    
    Pi_tmp(B_ind) = 1;
    I_ind = find(Pi_tmp==0);
    Pi_tmp = ~Pi;
    Pi_tmp(B1_ind) = 1;
    I1_ind = find(Pi_tmp==0);
    Pi_tmp = ~Pi;
    Pi_tmp(B2_ind) = 1;
    I2_ind = find(Pi_tmp==0);
	% find boundary tangential angles
    B1_ang = zeros(size(B1_ind));
    bbw = max(3,bw);
    nbp = length(B1_ind);
    Pi_tmp = zeros(size(Pi));
    Pi_tmp(B1_ind) = 1;
    oneside = (bbw-1)/2;
    if check_boundary
        figure;
        hold on
    end
    for i = 1:nbp
		localSize = 0;
		while localSize<2
        	[r, c] = ind2sub(size(Pi),B1_ind(i));
        	[ilist, localSize] = check(r,c,nx,ny,bbw);
			% find local boundary points in the subgrid
        	local_ind = find(Pi_tmp(ilist)>0);
			bbw = bbw+2;
		end
		% generate indices for the subgrid
        [rlocal, clocal] = ind2sub(localSize,local_ind);
        % WRONG!!! row is y, colum is x
        % WRONG!!! row is x, colum is y
        coeff = pca([rlocal-mean(rlocal), clocal-mean(clocal)]);
		% estimate the local slope with local boundary points
        B1_ang(i) = atan2(coeff(2,1),coeff(1,1));

        if check_boundary
            k = tan(B1_ang(i));
            if abs(k) > 1
                y = [c-oneside,c,c+oneside];
                x = 1/k*(y-c)+r;
            else
                x = [r-oneside,r,r+oneside];
                y = k*(x-r)+c;
            end
            plot(x,y,'-');
            plot(r,c,'*r', 'MarkerSize', 0.5);
        end
    end
    if check_boundary
        daspect([1,1,1]);
		print(gcf, '-loose', '-r2000', '-dpng', [prefix,'boundary_angle.png']);
		close(gcf);
    end
end
% return the indices of the boundary point's nearby subgrid (w x w) in the big grid
function [ilist, localSize] = check(ix,iy,nx,ny,w)
    oneside = (w-1)/2;
    lx = max([1,ix-oneside]);
    rx = min([nx,ix+oneside]);
    ly = max([1,iy-oneside]);
    ry = min([ny,iy+oneside]);
    [xlist, ylist] = ndgrid(lx:rx, ly:ry);
    ilist = sub2ind([nx,ny],xlist(:),ylist(:));
    localSize = [rx-lx,ry-ly]+1;
end
