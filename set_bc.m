function [B_ind, I_ind, B1_ind, I1_ind, B2_ind, I2_ind, B1_ang] = set_bc(Pi, bw, right_open, check_boundary, ENdir)
	% assuming horizontal symmetry
	right_most = 0;
	% the open boundary is to the right
    [nx, ny] = size(Pi);
    B_ind = zeros(nx*ny,1);
    B1_ind = zeros(nx*ny,1);
	if right_open
		% B2_ind includes the open boundary
    	B2_ind = zeros(nx*ny,1);
		k = 0;
	end
    i = 0;
    j = 0;
    for iy = 1:ny
        head = find(Pi(:,iy),bw,'first'); 
        tail = find(Pi(:,iy),bw,'last');
        if ~isempty(head)
			if right_most < head(end)
				right_most = head(end);
			end
			if ~right_open
            	B_ind(i+1:i+length(head)+length(tail)) = (iy-1)*nx + [head; tail];
            	i = i + length(head)+length(tail);
            	B1_ind(j+1:j+2) = (iy-1)*nx + [head(1); tail(end)];
            	j = j + 2;
			else
            	B_ind(i+1:i+length(head)) = (iy-1)*nx + head;
            	i = i + length(head);
            	B1_ind(j+1) = (iy-1)*nx + head(1);
            	j = j + 1;
            	B2_ind(k+1:k+2) = (iy-1)*nx + [head(1); tail(end)];
            	k = k + 2;
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
            B_ind(i+1:i+length(head)+length(tail)) = ([head tail]-1)'*nx + ix;
            B1_ind(j+1:j+2) = ([head(1) tail(end)]-1)'*nx + ix;
            i = i + length(head)+length(tail);
            j = j + 2;
			if right_open
            	B2_ind(k+1:k+2) = ([head(1) tail(end)]-1)'*nx + ix;
            	k = k + 2;
			end
        else
            assert(isempty(tail));
        end
    end
    B_ind = B_ind(1:i);
    B_ind = unique(B_ind);
    B1_ind = B1_ind(1:j);
    B1_ind = unique(B1_ind);
	if ~right_open
		B2_ind = B1_ind;
	else
    	B2_ind = B2_ind(1:k);
    	B2_ind = unique(B2_ind);
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
    nbp = length(B1_ind);
    Pi_tmp = zeros(size(Pi));
    Pi_tmp(B1_ind) = 1;
    mid = (bw+1)/2;
    oneside = (bw-1)/2;
    if check_boundary
        figure;
        hold on
    end
    for i = 1:nbp
        [r, c] = ind2sub(size(Pi),B1_ind(i));
        [ilist, localSize] = check(r,c,nx,ny,bw);
		% find local boundary points in the subgrid
        local_ind = find(Pi_tmp(ilist)>0);
		% generate indices for the subgrid
        [rlocal, clocal] = ind2sub(localSize,local_ind);
        % WRONG!!! row is y, colum is x
        coeff = pca([clocal-mean(clocal), rlocal-mean(rlocal)]);
		% estimate the local slope with local boundary points
        B1_ang(i) = atan2(coeff(2,1),coeff(1,1));

        if check_boundary
            k = tan(B1_ang(i));
            if abs(k) > 1
                y = [r-oneside,r,r+oneside];
                x = 1/k*(y-r)+c;
            else
                x = [c-oneside,c,c+oneside];
                y = k*(x-c)+r;
            end
            plot(x,y,'-');
            plot(c,r,'*r', 'MarkerSize', 0.5);
        end
    end
    if check_boundary
        daspect([1,1,1]);
		print(gcf, '-loose', '-r900', '-dpng', [ENdir,'/boundary_angle.png']);
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
