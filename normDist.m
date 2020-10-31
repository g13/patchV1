% find min distance to border, draw line, pick the opposite border, calculate the normalized pos of pinw inbetween.
function d = normDist(ORpinw, ODborders, lm, fign)
    if nargin < 4
        fign = 0;
    end
    tol = pi/8;
    n = size(ORpinw,1);
    m = size(ODborders,1);
    d = zeros(n,1);
	if m == 0
		return;
	end
    od = d;
    bpx = d;
    bpy = d;
    dist = ENsqdist(ORpinw, ODborders);
    [ud, ind] = min(dist,[],2);
    row = 1:n;
    dist(sub2ind([n,m], row', ind)) = inf;
    [~, ind2] = min(dist,[],2);
    for i = 1:n
        if abs(ind2(i)-ind(i)) == 1
            [bpx(i), bpy(i)] = p2p_line(ODborders(ind(i),:), ODborders(ind2(i),:), [ORpinw(i,1),ORpinw(i,2)]);
            ud(i) = (bpx(i)-ORpinw(i,1))^2 + (bpy(i)-ORpinw(i,2))^2;
        else
            bpx(i) = ODborders(ind(i),1);
            bpy(i) = ODborders(ind(i),2);
        end
    end
    ud = sqrt(ud);
    dy0 = bpy - ORpinw(:,2);
    dx0 = bpx - ORpinw(:,1);
    theta = atan2(dy0,dx0);

    for i=1:n
        y0 = ORpinw(i,2);
        x0 = ORpinw(i,1);
        dy = ODborders(:,2)-y0;
        dx = ODborders(:,1)-x0;
        dtheta = abs(atan2(dy,dx) - theta(i));
        pick = dtheta > pi;
        dtheta(pick) = abs(2*pi - dtheta(pick));
        pickOtherSide = dtheta > pi - tol;
        ipick = find(pickOtherSide);
        oppositeBorders = ODborders(pickOtherSide,:);
        if ~isempty(oppositeBorders)
            dist = dist2(oppositeBorders, x0, y0);
            [od(i), ond] = min(dist);
            dist(ond) = inf;
            [~, ond2] = min(dist);
            if abs(ipick(ond)-ipick(ond2)) == 1
                [box, boy] = p2p_line(oppositeBorders(ond,:),...
                    oppositeBorders(ond2,:), [ORpinw(i,1),ORpinw(i,2)]);
                od(i) = (box-ORpinw(i,1))^2 + (boy-ORpinw(i,2))^2;
            else
                boy = oppositeBorders(ond,2);
                box = oppositeBorders(ond,1);
            end
            od(i) = sqrt(od(i));
            ody = boy - y0;
            odx = box - x0;
            %otheta = abs(atan2(ody,odx) - theta(i));
            %if otheta > pi
            %    otheta = abs(2*pi - otheta);
            %end
			%if otheta <= pi/2
			%	otheta
			%	theta(i)
			%	odx
			%	ody
            %	assert(otheta > pi/2);
			%end
            if fign > 0
                figure(fign);
                hold on
                plot([box,bpx(i)],[boy,bpy(i)],':k');
            end
        else
            od(i) = max(ud(i),lm);
        end
    end
    d = min([ud,od],[],2)./(od+ud);
    assert(sum(d>0.5+10*eps) == 0);
end

function [x, y] = p2p_line(p1, p2, p)
    if p2(1) == p1(1)
        x = p1(1);
        y = p(2);
    else
        k = (p2(2)-p1(2))/(p2(1)-p1(1));
        x = (p2(1)*k^2+k*(p(2)-p2(2))+p(1))/(1+k^2);
        y = k*(x-p2(1)) + p2(2);
        x = min([x,max([p1(1),p2(1)])]);
        x = max([x,min([p1(1),p2(1)])]);
        y = min([y,max([p1(2),p2(2)])]);
        y = max([y,min([p1(2),p2(2)])]);
    end
end

function d = dist2(xy, x0, y0)
    d = (xy(:,1)-x0).^2 + (xy(:,2)-y0).^2;
end
