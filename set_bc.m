function [B_ind, I_ind, I1_ind, B1_ind, B1_ang] = set_bc(Pi, bw, check_boundary)
    [nx, ny] = size(Pi);
    blank = find(Pi==0);
    B_ind = zeros(nx*ny,1);
    B1_ind = zeros(nx*ny,1);
    i = 0;
    j = 0;
    for ix = 1:nx
        head = find(Pi(ix,:),bw,'first');
        tail = find(Pi(ix,:),bw,'last');
        if ~isempty(head)    
            B_ind(i+1:i+length(head)+length(tail)) = ([head tail]-1)'*nx + ix;
            B1_ind(j+1:j+2) = ([head(1) tail(end)]-1)'*nx + ix;
            i = i + length(head)+length(tail);
            j = j + 2;
        else
            assert(isempty(tail));
        end
    end
    for iy = 1:ny
        head = find(Pi(:,iy),bw,'first'); 
        tail = find(Pi(:,iy),bw,'last');
        if ~isempty(head)
            B_ind(i+1:i+length(head)+length(tail)) = (iy-1)*nx + [head; tail];
            B1_ind(j+1:j+2) = (iy-1)*nx + [head(1); tail(end)];
            i = i + length(head)+length(tail);
            j = j + 2;
        else
            assert(isempty(tail));
        end
    end
    B_ind = B_ind(1:i);
    B_ind = unique(B_ind);
    B1_ind = B1_ind(1:j);
    B1_ind = unique(B1_ind);
    Pi_tmp = Pi;    
    Pi_tmp([blank; B_ind]) = 0;
    I_ind = find(Pi_tmp>0);
    Pi_tmp = ~Pi;
    Pi_tmp(B1_ind) = 1;
    I1_ind = find(Pi_tmp==0);
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
        local_ind = find(Pi_tmp(ilist)>0);
        assert(length(local_ind) >= bw);
        [rlocal, clocal] = ind2sub(localSize,local_ind);
        % WRONG!!! row is y, colum is x
        coeff = pca([clocal-mean(clocal), rlocal-mean(rlocal)]);
        B1_ang(i) = atan2(coeff(2,1),coeff(1,1));
%         B1_ang(i) = atan2(max(clocal) - min(clocal), max(rlocal) - min(rlocal));

        if check_boundary
            %figure;
            %hold on
            k = tan(B1_ang(i));
            if abs(k) > 1
                y = [r-oneside,r,r+oneside];
                x = 1/k*(y-r)+c;
            else
                x = [c-oneside,c,c+oneside];
                y = k*(x-c)+r;
            end
            plot(x,y,'-');
            plot(c,r,'*r');
            %plot(rlocal+r-mid,clocal+c-mid,'ob');
            %plot(mean(rlocal+r-mid),mean(clocal+c-mid),'*b');
            %daspect([1,1,1]);
        end
    end
    if check_boundary
        daspect([1,1,1]);
    end
end
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
