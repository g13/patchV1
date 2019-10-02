% ENplot(G,bc,mu,plottype,v[,T,Pi,fg,ax]) Plot elastic net
%
% In:
%   G: list of grid lengths.
%   bc: string containing the type of boundary conditions: 'periodic'
%      or 'nonperiodic'.
%   mu: MxD matrix of M D-dimensional row vectors containing the
%      reference vectors.
%   plottype: one of the following:
%    'proj_cart': projection of the coordinates given by `v' in Cartesian
%      coordinates. `v' must be a 2x1 or 3x1 vector (no variables range
%      are necessary). The variables are considered in the order [x;y] (x
%      horizontal, y vertical) in 2D and [x;y;z] (z vertical) in 3D.
%      Utility for cortical maps: to plot the retinotopic coordinates.
%    'proj_polar': projection of the coordinates given by `v' in polar or
%      cylindrical coordinates. Useful for periodic variables. `v' must
%      be as in 'proj_cart'.
%    'img': greyscale image. `v' must be a 1x3 vector; v(1,2) appears
%      black and v(1,3) white.
%      Utility for cortical maps: to plot the ocular dominance or the
%      orientation selectivity (orientation tuning strength).
%    'img_per': colour image of a periodic variable, where the colour hue
%      (taken from a colour wheel) indicates the angle value and the
%      colour brightness (from 0 to a maximum) indicates the modulus. `v'
%      must be a matrix of 1x3 or 2x3, where v(1,1) indicates the angle,
%      with range [v(1,2) v(1,3)] (e.g. -pi to pi) and, if present,
%      v(2,1) indicates the modulus, with range [v(2,2) v(2,3)]
%      (typically [0 max]).
%      Utility for cortical maps: to plot the orientation preference with
%      ("polar map") or without ("angle map") the tuning strength.
%    'contour': contour plot of the variables given by `v'. `v' must be a
%      ?x3 matrix; the contour lines are equispaced in the ranges.
%      Utility for cortical maps: to plot the ocular dominance columns.
%    'contour_per': like 'contour' but for periodic variables.
%      Utility for cortical maps: to plot the isoorientation lines.
%   v: ?x3 matrix where each row contains the index and range of a
%      reference vector variable to be plotted. See `plottype' for details.
%   T: NxD matrix of N D-dimensional row vectors (training set of
%      "cities" or "stimuli vectors" in D dimensions). If T is not given,
%      or if it is empty [], only the reference vectors mu are plot.
%      It is ignored for plot types other than 'proj_cart' or 'proj_polar'.
%   Pi: 1xM list of mixing proportions (a priori probability of each grid
%      point, or centroid). They must be real numbers in [0,1] and add to 1,
%      although here we consider Pi as a logical vector where 0/non-0
%      disables/enables a centroid, respectively, from being plotted,
%      which is useful to define non-rectangular cortical shapes in 2D.
%      Default: Pi(m) = 1/M (equiprobable Gaussian mixture).
%   fg: Matlab-style specification of the figure position and size, i.e.,
%      an array [left, bottom, width, height] where `left' and `bottom'
%      define the distance from the lower-left corner of the screen to
%      the lower-left corner of the figure window and `width' and
%      `height' define the dimensions of the window. If fg is empty [],
%      no figure formatting is done; if fg is not given, it is
%      automatically derived to enclose the data; if only [left, bottom]
%      is given, it is used for the position but the size is derived; and
%      if only [left, bottom, l] is given, it is used for the position
%      and the size is derived so that l = largest dimension (i.e., both
%      width, length <= l).
%   ax: Matlab-style specification of the axis limits, i.e., an array
%      [ax1min ax1max ... axDmin axDmax] of the minimal and maximal value
%      for the axis of each variable. If ax is empty [], no axis formatting
%      is done, and the aspect ratios are not enforced; if ax is not
%      given, it is automatically derived to enclose the data, and 1:1
%      aspect ratios are enforced.
%
% Notes:
%
% - Axes orientations in the figure:
%   Plots in city space ('proj_cart' or 'proj_polar'): for 3D views, the
%   axes are drawn; for 2D views, the axes are not drawn and correspond to:
%   . Variable v(1): horizontal, growing rightwards (x axis).
%   . Variable v(2): vertical, growing upwards (y axis).
%   Plots in grid space ('img*', 'contour*'): the axes are not drawn and
%   the grid directions are as follows (el_net flips up-down this setup):
%   . Direction G(1): horizontal, growing rightwards.
%   . Direction G(2): vertical, growing downwards.
%
% - Except for 'proj_polar', ENplot plots to the figure without clearing
%   it. This is useful, for example, to superpose contour lines of
%   different variables, or to superpose different nets on the same cities.
%   If you want to clear the figure, use "cla" (not "clf") so that the
%   axes remain (and so the viewpoint isn't lost).
%
% - To add a title to a plot, always force 'Visible' to 'on', e.g.:
%     ENplot(G,bc,mu,'proj_cart',[1;2],T);
%     title('Retinotopic map','Visible','on');
%   This is because 'proj*' plots don't draw the axes.
%
% - When plotting the OD/OR contours from ENV1replay2, if the net
%   size is small (e.g. 10x10) then there is a large border around the
%   contour plot. This is not a bug: the contour plot is defined at the
%   centres of the image pixels, so the border is the half-pixel at the
%   map boundaries.
%
% Examples of use in the context of the travelling salesman problem (1D,
% 2D net):
%
%   ENplot(G,bc,mu,'proj_cart',(1:3)',T)
%     plot variables 1, 2 and 3 along the axes x, y and z, respectively;
%     plot the cities too.
%   ENplot(G,bc,mu,'proj_cart',(1:2)')
%     plot variables 1 and 2 along the axes x and y, respectively;
%     don't plot the cities.
%
% Examples of use in the context of cortical maps (2D net): assume that
% the indices in the reference vectors mu(:,d) correspond to x-retinotopy
% (1), y-retinotopy (2), ocular dominance (3), x-orientation (4) and
% y-orientation (5):
%
%   % Augment the reference vectors with orientation preference (6) in
%   % [-pi/2,pi/2] and orientation tuning strength (7):
%   mu = [mu zeros(size(mu,1),2)];
%   [mu(:,end-1) mu(:,end)] = cart2pol_sin(mu(:,4),mu(:,5));
%   mu(:,end-1) = mu(:,end-1) / 2;
%
%   ENplot(G,bc,mu,'proj_cart',[1;2],T) --> retinotopic map
%     plot retinotopic variables (x-retinotopic in the x axis,
%     y-retinotopic in the y axis), including the "cities".
%   ENplot(G,bc,mu,'proj_polar',[4;5])
%     plot orientation preference as the polar angle and tuning strength as
%     the modulus.
%   ENplot(G,bc,mu,'img',[3 -0.1 0.1]) --> ocular dominance map
%     plot ocular dominance as greyscale, where -0.1 is black and 0.1 is
%     white.
%   ENplot(G,bc,mu,'img_per',[6 -pi/2 pi/2]) --> orientation angle map
%     plot orientation preference as colour hue.
%   ENplot(G,bc,mu,'img_per',[6 -pi/2 pi/2;7 0 0.5]) --> orientation polar map
%     plot orientation preference as colour hue and orientation tuning
%     strength as brightness, where 0 is black and 0.5 maximum brightness.
%   ENplot(G,bc,mu,'contour',[3 -0.1 0.1;6 -pi/2 pi/2])
%     plot contours of ocular dominance and orientation preference (with
%     fake discontinuities).
%   ENplot(G,bc,mu,'contour_per',[6 -pi/2 pi/2])
%     plot contours of orientation preference.
%   ENplot(G,bc,mu,'img',[7 0 0.5]) --> selectivity map
%     plot orientation tuning strength as brightness, where 0 is black
%     and 0.5 maximum brightness.
%
% To do (Matlab developers should have done these...):
%
% - For 'proj_polar' in 3D, create polar3 function to plot cylindrical
%   coordinates (Matlab plots only polar coordinates).
%
% - For 'contour_per', plot each contour line with its corresponding
%   colour (in 'img_per'). For 'contour' this can be done simply by not
%   forcing a colour and linestyle in the contour plot, and setting the
%   colormap to hsv(256). But not for 'contour_per', since here ENsplice
%   causes the upper half of the colormap to be unused.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% See also ENscrinfo, ENreplay, ENtrset, ENgridtopo, ENtr_ann.

% Copyright (c) 2001 by Miguel A. Carreira-Perpinan

function myplot(G,bc,mu,plottype,v,T,Pi,fg,ax)
L = length(G);                  % Net dimensionality
if L == 1 G = [G 1]; end        % So that both L=1 and L=2 work
[M,D] = size(mu);

% Colours and markers
mucol = 'r'; mushape = 's'; musize = min(5,500/M);      % Centroids
Tcol = 'b'; Tshape = 's'; Tsize = musize*8/5;           % Cities

if ischar(plottype)
    % Fill in a structure with default values; depending on the plot type,
    % some of the values may not be necessary. The values follow Matlab's
    % conventions for plots (line types, etc.).
    plotv = struct('type',plottype,...            % Plot type
        'cmap',zeros(256,3),...        % Colormap for 'img*'
        'line',struct('lsty','?',...   % LineStyle
        'lcol','?',...   % LineColor
        'lwid',0,...     % LineWidth
        'msty','?',...   % MarkerStyle
        'msiz',0,...     % MarkerSize
        'mecol','?',...  % MarkerEdgeColor
        'mfcol','?',...  % MarkerFaceColor
        'num',0));       % No. lines for 'contour*'
    switch plotv.type
        case {'proj_cart','proj_polar'}
            % Centroids
            plotv.line(1).lsty = '-';
            plotv.line(1).lcol = 'r';
            plotv.line(1).lwid = 1;
            plotv.line(1).msty = 's';
            plotv.line(1).msiz = min(5,500/M);
            plotv.line(1).mecol = 'r';
            plotv.line(1).mfcol = 'r';
            plotv.line(1).num = 1;                      % Unused
            % Cities
            plotv.line(2).lsty = 'none';
            plotv.line(2).lcol = 'b';
            plotv.line(2).lwid = 1;
            plotv.line(2).msty = 's';
            plotv.line(2).msiz = plotv.line(1).msiz*8/5;
            plotv.line(2).mecol = 'b';
            plotv.line(2).mfcol = 'none';
            plotv.line(2).num = 1;                      % Unused
        case 'img'
            plotv.cmap = gray(256);
        case 'img_per'
            plotv.cmap = hsv(256);
        case {'contour','contour_per'}
            if strcmp(plotv.type,'contour_per')
                cols = 'brgmyck';                         % Cycle over these colours
            else
                cols = 'kbrgmyc';                         % Cycle over these colours
            end
            plotv.line(1).lsty = '-';
            plotv.line(1).lcol = cols(1);
            plotv.line(1).lwid = 1;
            plotv.line(1).msty = 'none';
            plotv.line(1).msiz = 1;
            plotv.line(1).mecol = 'none';
            plotv.line(1).mfcol = 'none';
            plotv.line(1).num = linspace(v(1,2),v(1,3),17);
            %    plotv.line(1).num = 10;
            for i = 2:size(v,1)
                plotv.line(i) = plotv.line(1);
                plotv.line(i).lcol = cols(mod(i-1,length(cols))+1);
            end
    end
else
    plotv = plottype;
end

[scw,sch,fgt,fgb,wnt,wnb] = ENscrinfo;			% Screen geometry

% Create a figure if there aren't any
curfig = get(0,'CurrentFigure');
if isempty(curfig)
    figure(1);
    set(figure(1),'Position',[1 50 10 10],'PaperPositionMode','auto');
end

% Set view for 3D plots
if (strcmp(plotv.type,'proj_cart') | strcmp(plotv.type,'proj_polar')) & ...
        size(v,1) == 3
    if isempty(curfig) | isempty(get(curfig,'CurrentAxes'))
        % Use a reasonable default view
        az = -15; el = 15;
        %    az = 322.5; el = 30;
    else
        % Preserve view (so that the user can interactively change it with
        % the "rotate 3d" button)
        [az,el] = view;
    end
end

% Figure position, largest dimension (not counting borders) and borders
% (in pixels).
% Note: Matlab plots the title just above the graph; it would be nicer if
% it centred it vertically on the blank space above the graph.
if ~exist('fg','var') | isempty(fg)
    fgp = get(gcf,'Position'); fgp = fgp(1:2);		% Position
else
    fgp = fg(1:2);
end
if exist('fg','var') & (length(fg) > 2)
    fgl = fg(3);
else
    fgl = floor(min(scw,sch)/2);				% Largest dimension
end
if exist('fg','var') && (length(fg) < 4)
    clear fg;
end

% Fast, flash-free rendering. Notes:
% - The 'OpenGL' renderer doesn't seem any faster than the 'painters'.
% - For 'proj*' plots, 'painters' is much faster than 'zbuffer' (which we
%   don't need because we don't have hidden objects).
set(gcf,'DoubleBuffer','on','Renderer','painters');
set(gca,'SortMethod','childorder'); % updated by Wei May 31 2019
% original: set(gca,'DrawMode','fast');

% Argument defaults ----------------------------------------------------
if ~exist('T','var') || isempty(T) T = NaN(1,D); end

if ~exist('Pi','var') || isempty(Pi)
    zPi = [];
else
    zPi = find(Pi==0);
    %  mu(zPi,:) = NaN;	% Disabled centroids aren't plotted or involved in axes
end

if ~exist('ax','var') || isempty(ax)
    if strcmp(plotv.type,'proj_cart') || strcmp(plotv.type,'proj_polar')
        % Axes: tightly given by the training set and reference vectors
        tmp = cat(1,T(:,v(:,1)),mu(:,v(:,1)));
        ax = [min(tmp); max(tmp)]; ax = ax(:)';
    else
        % Axes: given by the grid
        ax = [-1 1 -1 1]/2 + [1 G(1) 1 G(2)];
    end
end
% Axis ratio
if ~isempty(ax)
    axR = (ax(2)-ax(1))/(ax(4)-ax(3));
else
    tmp = axis;
    axR = (tmp(2)-tmp(1))/(tmp(4)-tmp(3));
end
% For 1D nets, limit axis ratio to avoid too thin images
switch plotv.type
    case {'img','img_per','contour','contour_per'}
        if G(2) == 1
            axR = min(axR,50);
        elseif G(1) == 1
            axR = max(axR,1/50);
        end
end

if ~exist('fg','var')
    % Figure size: given by axes, enforcing 1:1 aspect ratio, plus some
    % space for the title and margins.
    if strcmp(plotv.type,'proj_cart') & (size(v,1) == 3)
        fg = [fgp fgt*[0 1]+fgl*[1 1]];
    else
        fg = [fgp 2*fgb*[1 1]+fgt*[0 1]+fgl*[min(axR,1) min(1/axR,1)]];
    end
end

if ~isempty(fg) && ~isempty(ax)
    % axfg fits axes as large as possible inside the figure, respecting the
    % margins and aspect ratio.
    axfg = [fgb/fg(3) fgb/fg(4) 1-2*fgb/fg(3) 1-(fgt+2*fgb)/fg(4)];
    fgR = (fg(3)-2*fgb)/(fg(4)-2*fgb-fgt);                % Usable figure ratio
    if fgR < axR                                          % Respect aspect ratio
        axfg = axfg + [0 1/2 0 -1] * ((1-fgR/axR)*axfg(4));
    else
        axfg = axfg + [1/2 0 -1 0] * ((1-axR/fgR)*axfg(3));
    end
end
% ----------------------------------------------------------------------

% PLOT
hold on;
switch plotv.type
    
    % ----------------------------------------------------------------------
    case {'proj_cart','proj_polar'}
        
        % Data preparation
        mu(zPi,:) = NaN;			% Disabled centroids aren't plotted
        if strcmp(plotv.type,'proj_cart')
            Mux = reshape(mu(:,v(:,1)),[G size(v,1)]);
        else
            [T_t T_r] = cart2pol(T(:,v(1,1)),T(:,v(2,1)));
            [mu_t mu_r] = cart2pol(mu(:,v(1,1)),mu(:,v(2,1)));
            Mux = reshape([mu_t mu_r mu(:,v(3:end,1))],[G size(v,1)]);
        end
        
        if L == 2 Muy = Mux; end
        
        if strcmp(bc,'periodic')              % Closed tour: ring (1D) or torus (2D)
            Mux = cat(1,Mux,Mux(1,:,:));
            if L == 2 Muy = cat(2,Muy,Muy(:,1,:)); end
        end
        
        % Plot commands
        if strcmp(plotv.type,'proj_cart')
            if size(v,1) == 2
                plotcmdT = 'plot(T(:,v(1,1)),T(:,v(2,1))';
                plotcmdMx = 'plot(Mux(:,:,1),Mux(:,:,2)';
                plotcmdMy = 'plot(Muy(:,:,1)'',Muy(:,:,2)''';
            else
                plotcmdT = 'plot3(T(:,v(1,1)),T(:,v(2,1)),T(:,v(3,1))';
                plotcmdMx = 'plot3(Mux(:,:,1),Mux(:,:,2),Mux(:,:,3)';
                plotcmdMy = 'plot3(Muy(:,:,1)'',Muy(:,:,2)'',Muy(:,:,3)''';
            end
        else
            if size(v,1) == 2
                plotcmdT = 'hold off; tmp=polar(T_t,T_r); hold on;set(tmp';
                plotcmdMx = 'tmp=polar(Mux(:,:,1),Mux(:,:,2));set(tmp';
                plotcmdMy = 'tmp=polar(Muy(:,:,1)'',Muy(:,:,2)'');set(tmp';
            else
                % "polar3": plot cylindrical coordinates in Matlab. How to do it?
                plotcmdT = 'hold off; tmp=polar3(T_t,T_r,T(:,v(3,1))); hold on;set(tmp';
                plotcmdMx = 'tmp=polar3(Mux(:,:,1),Mux(:,:,2),Mux(:,:,3));set(tmp';
                plotcmdMy = 'tmp=polar3(Muy(:,:,1)'',Muy(:,:,2)'',Muy(:,:,3)'');set(tmp';
            end
        end
        plotattrT = [',''LineStyle'',plotv.line(2).lsty,'...
            '''Color'',plotv.line(2).lcol,'...
            '''LineWidth'',plotv.line(2).lwid,'...
            '''Marker'',plotv.line(2).msty,'...
            '''MarkerSize'',plotv.line(2).msiz,'...
            '''MarkerEdgeColor'',plotv.line(2).mecol,'...
            '''MarkerFaceColor'',plotv.line(2).mfcol);'];
        plotattrMx = [',''LineStyle'',plotv.line(1).lsty,'...
            '''Color'',plotv.line(1).lcol,'...
            '''LineWidth'',plotv.line(1).lwid,'...
            '''Marker'',plotv.line(1).msty,'...
            '''MarkerSize'',plotv.line(1).msiz,'...
            '''MarkerEdgeColor'',plotv.line(1).mecol,'...
            '''MarkerFaceColor'',plotv.line(1).mfcol);'];
        plotattrMy = [',''LineStyle'',plotv.line(1).lsty,'...
            '''Color'',plotv.line(1).lcol,'...
            '''LineWidth'',plotv.line(1).lwid,'...
            '''Marker'',''none'','...
            '''MarkerSize'',plotv.line(1).msiz,'...
            '''MarkerEdgeColor'',plotv.line(1).mecol,'...
            '''MarkerFaceColor'',plotv.line(1).mfcol);'];
        
        % Plot cities
        eval([plotcmdT plotattrT]);
        % Plot reference vectors and edges between them (grid rows)
        eval([plotcmdMx plotattrMx]);
        % Plot edges between them (grid columns)
        if L == 2
            eval([plotcmdMy plotattrMy]);
        end
        % ----------------------------------------------------------------------
        
        % ----------------------------------------------------------------------
    case {'img','img_per'}
        cmap = plotv.cmap; colormap(cmap);
        tmp1 = ENnormalise(mu(:,v(1,1)),v(1,2:3),[1 size(cmap,1)]);
        if strcmp(plotv.type,'img_per') && (size(v,1) == 2)
            tmp2 = ENnormalise(mu(:,v(2,1)),v(2,2:3),[0 1]);
            % Crap Matlab's rot90 only works for 2D matrices, so I have to rotate
            % each colour layer (R, G, B) separately:
            tmp3 = flipdim(reshape(...
                cmap(round(tmp1),:) .* repmat(tmp2,1,size(cmap,2)),...
                G(1),G(2),size(cmap,2)),2);
            image(cat(3,rot90(tmp3(:,:,1)),rot90(tmp3(:,:,2)),rot90(tmp3(:,:,3))));
        else
            image(rot90(fliplr(reshape(tmp1,G(1),G(2)))));
        end
        % ----------------------------------------------------------------------
        
        % ----------------------------------------------------------------------
    case {'contour','contour_per'}
        mu(zPi,:) = NaN;			% Disabled centroids aren't plotted
        for i = 1:size(v,1)
            if strcmp(plotv.type,'contour_per')
                tmp1 = ENsplice(mu(:,v(i,1)),v(i,2:3));
				[min(tmp1), max(tmp1)]
            else
                tmp1 = mu(:,v(i,1));
            end
            [tmp3 tmp2] = contour(rot90(fliplr(reshape(tmp1,G(1),G(2)))),...
                plotv.line(i).num,plotv.line(i).lsty);
            set(tmp2,'LineStyle',plotv.line(i).lsty,...
                'Color',plotv.line(i).lcol,...
                'LineWidth',plotv.line(i).lwid);
            % Commented out Aug 28, 2012 to prevent this Matlab error:
            %   There is no 'MarkerSize' property in the 'contourgroup' class.
            % $$$     set(tmp2,'LineStyle',plotv.line(i).lsty,...
            % $$$              'Color',plotv.line(i).lcol,...
            % $$$              'LineWidth',plotv.line(i).lwid,...
            % $$$              'Marker',plotv.line(i).msty,...
            % $$$              'MarkerSize',plotv.line(i).msiz,...
            % $$$              'MarkerEdgeColor',plotv.line(i).mecol,...
            % $$$              'MarkerFaceColor',plotv.line(i).mfcol);
        end
        % ----------------------------------------------------------------------
        
end
hold off;

% AXES
if ~isempty(ax)
    axis(ax);
    if strcmp(plotv.type,'proj_cart') | strcmp(plotv.type,'proj_polar')
        switch size(v,1)
            case 2
                set(gca,'Visible','off');
            case 3
                set(gca,'Visible','on','Box','on','DataAspectRatio',[1 1 1]);
                view(az,el);
        end
    else
        set(gca,'XTick',[],'YTick',[],'XDir','normal','YDir','reverse','Box','on');
        % Matlab doesn't seem to draw the box for images, so I have to do it:
        hold on;
        plot([ax(1) ax(2) ax(2) ax(1) ax(1)],[ax(3) ax(3) ax(4) ax(4) ax(3)],'k-');
        hold off;
    end
end

% FIGURE
if ~isempty(fg)
    if strcmp(plotv.type,'proj_cart') && (size(v,1) == 3)
        %set(gcf,'Position',fg,'Color','w','MenuBar','figure',...
        %    'PaperPositionMode','auto');
        set(gcf,'Position',fg,'Color','w',...
            'PaperPositionMode','auto');
    else
        %set(gcf,'Position',fg,'Color','w','MenuBar','none',...
        %   'PaperPositionMode','auto');
        set(gcf,'Position',fg,'Color','w',...
            'PaperPositionMode','auto');
        if ~isempty(ax)
            set(gca,'Position',axfg);
        end;
    end
end

drawnow;


% y = ENnormalise(x,fromR,toR)
%
% Linearly transforms the values of x from the range fromR onto the range
% toR, clipping (setting to NaN) out-of-range values.
% Useful to plot x with a colormap of colour indices 1 to 256 (say):
%   y = ENnormalise(x,fromR,[1 256]);
%
% In:
%   x: vector or matrix of real numbers.
%   fromR: vector [a b] containing the range where the elements of x
%      should lie, i.e., a <= x <= b.
%   toR: vector [a b] containing the range where the elements of y
%      should lie, i.e., a <= y <= b.

function y = ENnormalise(x,fromR,toR)

y = (x - fromR(1)) * (toR(2) - toR(1)) / (fromR(2) - fromR(1)) + toR(1);

% Clip out-of-range values:
y(y > toR(2)) = toR(2); y(y < toR(1)) = toR(1);

% If I return NaN for out-of-range values, I can't do cmap(NaN,:):
% y((y > toR(2)) | (y < toR(1))) = NaN; % Perhaps change toR+eps*[-1 1]

% The same happens if x already contained NaN's (as when Pi has zero
% values), so we set those to the lower end of the range:
y(isnan(y)) = toR(1);


% y = ENsplice(x,range)
%
% This function is a trick to get correct contour lines of periodic
% functions (phase fields). Say the function is periodic in [-pi,pi];
% then there is an apparent discontinuity at + and -pi. Since Matlab's
% `contour' function internally uses a (linear) interpolation of the data
% x, this produces steep slopes around the apparent discontinuity which
% result in many closely spaced contour lines there (to see it, use
% ENplot with plottype 'contour' on periodic data).
%
% The trick consists of computing the contour lines of an altered x so
% that:
% - the ends + and -pi are spliced, to avoid the discontinuity;
% - the slope magnitude is the same as in the original x, to have the
%   same contour lines as the original x.
% The function below does this, but it misses the contour line at the
% turning point in the interval centre (0 for [+pi -pi]), as can be seen
% by applying it to a plane wave. One can recover this missing contour
% line by:
% - slightly enlarging the interval where x and ENsplice(x) coincide, or
% - plotting equispaced contour lines that don't include + and -pi/2,
%   e.g. -pi+0.1:pi/8:pi instead of -pi:pi/8:pi.
% However, for complex phase fields, such as the orientation maps of the
% cortex, the difference is small.
%
% In:
%   x: vector or matrix of real numbers periodic in `range'.
%   range: vector [a b] containing the periodicity range of the elements
%      of x, i.e., a <= x <= b but b joins with a, e.g. [-pi pi].

function y = ENsplice(x,range)

% If range = [a b] then y = x for [a (a+b)/2] and y = a+b-x for ((a+b)/2 b].
y = x;
tmp = find(x > sum(range)/2);
y(tmp) = sum(range) - x(tmp);

