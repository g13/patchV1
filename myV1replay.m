% ENV1replay2(G,bc,ENlist,v[,whichones,T,Pi,murange,tens,opt,figlist])
% Replay elastic net sequence and optionally create movie for the case of
% 2D cortex in stimulus space (VFx,VFy,OD,ORx,ORy) where VF = visual
% field, OD = ocular dominance and OR = orientation.
%
% The following plots can be produced:
% - Figure 1: ocular dominance map.
% - Figure 2: OD-OR contours.
% - Figure 3: orientation angle map.
% - Figure 4: orientation polar map.
% - Figure 5: retinotopy (VFx,VFy) in Cartesian coordinates.
% - Figure 6: orientation (ORx,ORy) in polar coordinates.
% - Figure 7: orientation selectivity (ORr) map.
% - Figure 8: retinotopy + OD (VFx,VFy,OD) in Cartesian coordinates.    [SLOW]
% - Figure 9: orientation + OD (ORx,ORy,OD) in polar coordinates.       [SLOW]
% - Figure 10: retinotopy + ORr (VFx,VFy,ORr) in Cartesian coordinates. [SLOW]
% - Figure 11: tension (not including the beta/2 factor).
% - Figure 12: colorbar for fig. 11.
% - Figure 100: objective function value.
% - Figure 101: computation time.
% Other combinations of variables are not so interesting.
%
% This file is a bespoke version of ENreplay.m.
%
% In:
%   G: list of grid lengths.
%   bc: string containing the type of boundary conditions: 'periodic'
%      or 'nonperiodic'.
%   ENlist: the sequence of elastic nets, as created by ENtr_ann. It is a
%      structure array with fields mu and stats, where stats is a structure
%      array with fields K, E, time, cpu, code and it.
%   v: 5x3 matrix where each row contains the index and range of a
%      reference vector variable to be plotted. The rows are assumed to
%      mean the following:
%      1. Retinotopy along x (VFx).
%      2. Retinotopy along y (VFy).
%      3. Ocular dominance value (OD).
%      4. Orientation x-value (ORx).
%      5. Orientation y-value (ORy).
%   whichones: list of indices in ENlist, indicating which nets to plot,
%      e.g. [1:5 10:length(ENlist)]. If it is a negative integer Z, then
%      the last Z nets are selected. Default: all. The following strings
%      are also admitted (case ignored):
%      - 'all': as a shortcut for 1:length(ENlist).
%      - 'first': as a shortcut for 1.
%      - 'last': as a shortcut for length(ENlist) or -1.
%   T: NxD matrix of N D-dimensional row vectors (training set of
%      "cities" or "stimuli vectors" in D dimensions). If T is not given
%      or if it is empty, only the reference vectors mu are plot (figures
%      1-2 only).
%   Pi: 1xM list of mixing proportions (a priori probability of each grid
%      point, or centroid). They must be real numbers in [0,1] and add to 1,
%      although here we consider Pi as a logical vector where 0/non-0
%      disables/enables a centroid, respectively, from being plotted,
%      which is useful to define non-rectangular cortical shapes in 2D.
%      Default: Pi(m) = 1/M (equiprobable Gaussian mixture).
%   murange: 2x(D+2) matrix containing the minimum (row 1) and maximum
%      (row 2) values of all the centroids (augmented with (theta,r)) in
%      ENlist.mu (necessary to calculate the correct axes). If empty or
%      not given, they will be calculated. It is intended to save time and
%      memory when repeatedly calling ENV1replay2 but could also be used
%      to force a range. Default: [].
%   tens: either the DD matrix computed by ENgridtopo, or a cell array
%      {DD,th} where th is a number in (0,1] indicating where to set the
%      maximal tension to be plotted in fig. 11 (with higher tensions
%      saturating the colormap). See the code for fig. 11 for more
%      details. Default for th: 0.1. This argument is only necessary for
%      fig. 11; if this is not plotted, use [] for tens.
%   opt: 'pause' (to wait for a keystroke between successive plots),
%      '*.avi' or '*.mpg' (to create an AVI or MPG movie, where * will be
%      the movie name) or '' (to do nothing). Default: ''.
%   figlist: list of numbers corresponding to the figures to plot. Default:
%      [1 2 3 5 6 7]. To plot all figures use 1:1000.
%
% Notes:
%
% - The position of all figures is given by that of figure 1. If figure 1
%   already exists, its position in the screen is respected.
%
% - Colorbars can be added with ENcolorbar.
%
% - Make sure the DISPLAY variable is correctly set; otherwise the
%   command get(0,'ScreenSize') silently returns [1 1 1 1] and an error
%   appears in an unexpected function line.
%
% - In fig. 2 (OD/OR contours), if the net size is small (e.g. 10x10) then
%   there is a large border around the contour plot. This is not a bug: the
%   contour plot is defined at the centres of the image pixels, so the
%   border is the half-pixel at the map boundaries.
%
% To do:
%
% - Figure 9 should show (ORx,ORy,OD) in cylindrical coordinates rather
%   in Cartesian coordinates. This requires writing a "polar3" function
%   (see myplot) which Matlab unfortunately lacks.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% See also ENV1driver2, ENreplay, ENtr_ann, myplot, ENtrset, ENgridtopo.

% Copyright (c) 2002 by Miguel A. Carreira-Perpinan

function myV1replay(G,bc,ENlist,v,whichones,T,Pi,murange,id,tens,opt,figlist)

L = length(G);                  % Net dimensionality
if L == 1, G = [G 1]; end        % So that both L=1 and L=2 work
[M,D] = size(ENlist(1).mu);

% Argument defaults
if ~exist('whichones','var') || isempty(whichones)
    whichones = 1:length(ENlist);
end
if ischar(whichones)
    switch lower(whichones)
        case 'all'
            whichones = 1:length(ENlist);
        case 'first'
            whichones = 1;
        case 'last'
            whichones = length(ENlist);
    end
elseif whichones < 0
    whichones = max(1,length(ENlist)+whichones+1):length(ENlist);
end

if ~exist('T','var') || isempty(T), T = NaN(1,D); end
% Augment T with (theta,r) values for OR
if isfield(id, 'OR')
    T = [T zeros(size(T,1),2)];
    [T(:,end-1), T(:,end)] = cart2pol(T(:,id.ORx),T(:,id.ORy));
    T(:,end-1) = T(:,end-1) / 2;
end

if ~exist('Pi','var') || isempty(Pi)
    Pi = ones(1,M)/M;
else
    Pi = Pi/sum(Pi);	% Make sure the mixing proportions are normalised
end

% Ranges for all variables
if ~exist('murange','var') || isempty(murange)
    all = cat(1,ENlist.mu);
    if isfield(id, 'ORx')
        all = [all zeros(size(all,1),2)];
        [all(:,end-1), all(:,end)] = cart2pol(all(:,id.ORx),all(:,id.ORy));
        all(:,end-1) = all(:,end-1) / 2;
        all = [all;T];
    end
else
    all = cat(1,T,murange);
end
all = [min(all);max(all)];

if exist('tens','var') && ~iscell(tens) tens = {tens,0.1}; end;

domovie = 0;
if ~exist('opt','var') || isempty(opt)
    opt = '';
elseif length(opt) > 4                                  % Check for movie name
    tmp = lower(opt(end-3:end));
    if strcmpi(tmp,'.avi')
        domovie = 1;
        if isunix
            compression = 'None';
        else
            % $$$       compression = 'RLE';
            compression = 'Cinepak';
        end
    elseif strcmpi(tmp,'.mpg')
        domovie = 2;
    end
    moviecounter = 1;
end

if ~exist('figlist','var') || isempty(figlist)
    figlist = [1 2 3 4 5 7];
end
% plotfig(i) = 1 means plot figure(i).
plotfig = false(1,1000); plotfig(figlist) = 1;

% Precompute axes and figure size, the same for the whole sequence.
[scw,sch,fgt,fgb,wnt,wnb] = ENscrinfo;			% Screen geometry
% Figure parameters:
fgl = floor(min(scw*2,sch*2)/3);				% Largest dimension

gridLim = [-1 1 -1 1]/2 + [1 G(1) 1 G(2)];
gridR = (gridLim(2)-gridLim(1))/(gridLim(4)-gridLim(3));                 % Axis ratio
% For 1D nets, limit axis ratio to avoid too thin images
if G(2) == 1
    gridR = min(gridR,50);
elseif G(1) == 1
    gridR = max(gridR,1/50);
end

% Get the position of figure 1 if it exists, otherwise use a default
% (lower left corner).
if ~ishandle(1)
    fg1 = [1 50 10 10];
else
    fg1 = get(figure(1),'Position');
end

% Figure 1: 'img' for OD.
fg1p = fg1(1:2);					% Position (fig. 1)
% Axes: given by the grid
ax1 = gridLim;
% Figure size
ax1R = gridR;
fg1 = [fg1p 2*fgb*[1 1]+fgt*[0 1]+fgl*[min(ax1R,1) min(1/ax1R,1)]];
if plotfig(1)
    set(figure(1),'Position',fg1,'PaperPositionMode','auto','Color','w',...
        'Name','Ocular dominance (OD) map',...
        'DoubleBuffer','on','Renderer','painters');
        %'DoubleBuffer','on','Renderer','painters','MenuBar','none');
end

% Figure 2: 'contour' for OD and OR.
fg2p = fg1p;                                            % Position (fig. 2)
% Raise it to avoid overlapping with fig. 1
fg2p(2) = fg2p(2) + fg1(4) + wnb + wnt;
ax2 = gridLim; ax2R = gridR;                                 % Axes (fig. 2)
fg2 = [fg2p fg1(3:4)];
if ~ishandle(2)
    if plotfig(2)
        set(figure(2),'Position',fg2,'PaperPositionMode','auto','Color','w',...
            'Name','Contours of OD and OR',...
            'DoubleBuffer','on','Renderer','painters');
            %'DoubleBuffer','on','Renderer','painters','MenuBar','none');
    end
end
% Contour lines properties for OD (see myplot)
ODplotv = struct('type','contour',...             % Plot type
    'cmap',zeros(256,3),...          % Colormap for 'img*'
    'line',struct('lsty','-',...     % LineStyle
    'lcol','k',...     % LineColor
    'lwid',2.5,...       % LineWidth
    'msty','none',...  % MarkerStyle
    'msiz',1,...       % MarkerSize
    'mecol','none',... % MarkerEdgeColor
    'mfcol','none',... % MarkerFaceColor
    'num',[1 1]*mean(v(id.OD,2:3))));% Single line
% For most nets, the values of OD are balanced for both eyes and so using a
% single line and letting Matlab decide its level gives practically the same
% lines as forcing its level to be the middle point between both eyes.
% However, sometimes the net is unbalanced (notably for the 4th-order
% stencil) and the contours are then wrong.
%                               'num',1));         % No. lines for 'contour*'
% Ditto OR
if isfield(id, 'OR')
    ORplotv = ODplotv;
    ORplotv.type = 'contour_per';
    ORplotv.line.lwid = 1;
    ORplotv.line.num = linspace(v(id.OR,2),v(id.OR,3),17);
    ORplotv.line.lcol = 'b';
end
% Figure 3: 'img_per' for OR (angle map).
fg3p = fg1p;                                            % Position (fig. 3)
% Shift it right to avoid overlapping with fig. 1
fg3p(1) = fg3p(1) + fg1(3) + 2*wnb;
ax3 = gridLim; ax3R = gridR;                                 % Axes (fig. 3)
fg3 = [fg3p fg1(3:4)];
if plotfig(3)
    set(figure(3),'Position',fg3,'PaperPositionMode','auto','Color','w',...
        'Name','Orientation (OR) angle map',...
        'DoubleBuffer','on','Renderer','painters');
        %'DoubleBuffer','on','Renderer','painters','MenuBar','none');
end

% Figure 4: 'img_per' for OR (polar map).
fg4p = fg3p;                                            % Position (fig. 4)
% Raise it to avoid overlapping with fig. 3
fg4p(2) = fg4p(2) + fg3(4) + wnb + wnt;
ax4 = gridLim; ax4R = gridR;                                 % Axes (fig. 4)
fg4 = [fg4p fg1(3:4)];
if plotfig(4)
    set(figure(4),'Position',fg4,'PaperPositionMode','auto','Color','w',...
        'Name','Orientation (OR) polar map',...
        'DoubleBuffer','on','Renderer','painters');
        %'DoubleBuffer','on','Renderer','painters','MenuBar','none');
end

% Figure 5: 'proj_cart' for (VFx,VFy).
fg5p = fg3p;                                            % Position (fig. 5)
% Shift it right to avoid overlapping with fig. 3
fg5p(1) = fg5p(1) + fg3(3) + 2*wnb;
if isfield(id,'VFx') && isfield(id, 'VFy')
    % Axes: tightly given by the training set and reference vectors
    ax5 = all(:,v([id.VFx,id.VFy],1)); ax5 = ax5(:)';
    % Figure size
    ax5R = (ax5(2)-ax5(1))/(ax5(4)-ax5(3));                 % Axis ratio
    fg5 = [fg5p 2*fgb*[1 1]+fgt*[0 1]+fgl*[min(ax5R,1) min(1/ax5R,1)]];
    if plotfig(5)
        set(figure(5),'Position',fg5,'PaperPositionMode','auto','Color','w',...
            'Name','Retinotopic map (VFx,VFy)',...
            'DoubleBuffer','on','Renderer','painters');
            %'DoubleBuffer','on','Renderer','painters','MenuBar','none');
        xlabel('VFx'); ylabel('VFy');
    end
end


% Plot properties for 'proj_cart' (see myplot)
proj_cart_plotv = struct('type','proj_cart',... % Plot type
    'cmap',zeros(256,3),...          % Colormap for 'img*'
    'line',struct('lsty','-',...     % LineStyle
    'lcol','r',...     % LineColor
    'lwid',1,...       % LineWidth
    'msty','s',...     % MarkerStyle
    'msiz',min(5,500/M),...    % MarkerSize
    'mecol','r',...    % MarkerEdgeColor
    'mfcol','r',...    % MarkerFaceColor
    'num',1));         % No. lines for 'contour*'
% Cities
proj_cart_plotv.line(2).lsty = 'none';
proj_cart_plotv.line(2).lcol = 'b';
proj_cart_plotv.line(2).lwid = 1;
proj_cart_plotv.line(2).msty = 's';
proj_cart_plotv.line(2).msiz = 5;
proj_cart_plotv.line(2).mecol = 'b';
proj_cart_plotv.line(2).mfcol = 'none';
proj_cart_plotv.line(2).num = 1;                        % Unused

% Figure 6: 'proj_polar' for (ORx,ORy).
fg6p = fg4p;                                            % Position (fig. 6)
% Shift it right to avoid overlapping with fig. 4
fg6p(1) = fg6p(1) + fg4(3) + 2*wnb;
if fg6p(2) < fg4p(2) fg6p(2) = fg4p(2); end
if isfield(id, 'ORx') && isfield(id, 'ORy')
    % Axes: tightly given by the training set and reference vectors
    ax6 = all(:,v([id.ORx,id.ORy],1)); ax6 = ax6(:)';
    % Figure size
    ax6R = (ax6(2)-ax6(1))/(ax6(4)-ax6(3));                 % Axis ratio
    fg6 = [fg6p 2*fgb*[1 1]+fgt*[0 1]+fgl*[min(ax6R,1) min(1/ax6R,1)]];
    if plotfig(6)
        set(figure(6),'Position',fg6,'PaperPositionMode','auto','Color','w',...
            'Name','Orientation map (ORx,ORy)',...
            'DoubleBuffer','on','Renderer','painters','MenuBar','none');
        xlabel('ORx'); ylabel('ORy');
    end
end
% Plot properties for 'proj_polar' (see myplot)
proj_polar_plotv = proj_cart_plotv;
proj_polar_plotv.type = 'proj_polar';

% Figure 7: 'img' for ORr (OR selectivity map).
% I position it exactly over figure 4 (the polar map).
fg7p = fg4p;                                            % Position (fig. 7)
ax7 = ax4; ax7R = ax4R;                                 % Axes (fig. 7)
fg7 = fg4;
if ~ishandle(7)
    if plotfig(7)
        set(figure(7),'Position',fg7,'PaperPositionMode','auto','Color','w',...
            'Name','Orientation (OR) selectivity map',...
            'DoubleBuffer','on','Renderer','painters','MenuBar','none');
    end
end

% Figure 8: 'proj_polar' for (VFx,VFy,OD).
% For this plot, I could also colour every dot with its orientation value
% colour and represent all maps at the same time; but there are too many
% dots for this to be useful.
fg8p = fg5p;                                            % Position (fig. 8)
if isfield(id,'VFx') && isfield(id, 'VFy') && isfield(id, 'OD')
    % Axes: tightly given by the training set and reference vectors
    ax8 = all(:,v([id.VFx,id.VFy,id.OD],1)); ax8 = ax8(:)';
    % Figure size
    fg8 = fg5;
    if plotfig(8)
        set(figure(8),'Position',fg8,'PaperPositionMode','auto','Color','w',...
            'Name','Retinotopic map + OD (VFx,VFy,OD)',...
            'DoubleBuffer','on','Renderer','painters');
        xlabel('VFx'); ylabel('VFy'); zlabel('OD');
        view(-15,15);
    end
end

% Figure 9: 'proj_polar' for (ORx,ORy,OD).
fg9p = fg6p;                                            % Position (fig. 9)
if isfield(id,'ORx') && isfield(id, 'ORy') && isfield(id, 'OD')
    % Axes: tightly given by the training set and reference vectors
    ax9 = all(:,v([id.ORx,id.ORy,id.OD],1)); ax9 = ax9(:)';
    % Figure size
    fg9 = fg6;
    if plotfig(9)
        set(figure(9),'Position',fg9,'PaperPositionMode','auto','Color','w',...
            'Name','Orientation map + OD (ORx,ORy,OD)',...
            'DoubleBuffer','on','Renderer','painters');
        xlabel('ORx'); ylabel('ORy'); zlabel('OD');
        view(-15,15);
    end
end
% Matlab has no 3D polar plot, so we use proj_cart_plotv here too.

% Figure 10: 'proj_polar' for (VFx,VFy,ORr).
% For this plot, I could also colour every dot with its orientation value
% colour and represent all maps at the same time; but there are too many
% dots for this to be useful.
fg10p = fg4p;                                            % Position (fig. 10)
if isfield(id,'VFx') && isfield(id, 'VFy') && isfield(id, 'ORr')
    % Axes: tightly given by the training set and reference vectors
    ax10 = all(:,v([id.VFx,id.VFy,id.ORr],1)); ax10 = ax10(:)';
    % Figure size
    fg10 = fg4;
    if plotfig(10)
        set(figure(10),'Position',fg10,'PaperPositionMode','auto','Color','w',...
            'Name','Retinotopic map + OD (VFx,VFy,ORr)',...
            'DoubleBuffer','on','Renderer','painters');
        xlabel('VFx'); ylabel('VFy'); zlabel('ORr');
        view(-15,15);
    end
end

% Figure 11: 'img' for the tension (not including the beta/2 factor) as a
%            function of centroid location.
% Figure 12: colorbar for fig. 11.
fg11p = fg2p;                                            % Position (fig. 11)
ax11 = ax1; ax11R = ax1R;                                % Axes (fig. 11)
fg11 = [fg11p fg1(3:4)];
if plotfig(11) && exist('tens','var')
    set(figure(11),'Position',fg11,'PaperPositionMode','auto','Color','w',...
        'Name','Tension map',...
        'DoubleBuffer','on','Renderer','painters','MenuBar','none');
    % Compute tension maps for all frames to normalise the tension range.
    % This assumes that DD is KxM with K an integer multiple of M (as is the
    % case for both 2D stencils and horizontal-vertical 1D stencils).
    % The tension cannot be computed from the S matrix (at least easily).
    DD = tens{1};
    DDmu2 = zeros(M,length(ENlist));
    for ENcounter = 1:length(ENlist)
        DDmu2(:,ENcounter) = ...
            sum(reshape(sum((DD*ENlist(ENcounter).mu).^2,2),M,size(DD,1)/M),2);
    end
    % DDmu2 has the following characteristics:
    % - For frame 1 (the roughly topographic, noisy initial net) most
    %   centroids have a tension much larger than for the next frames.
    % - For the remaining frames, the histogram of tension values develops
    %   a very long right tail, i.e., most centroids have a relatively low
    %   tension but a few centroids have much larger tension (and there is
    %   where the twists of the net occur).
    % To obtain a meaningful plot, we must choose a range that doesn't mask
    % the values of most pixels: we don't include the first frame in the range
    % calculation, and we then set the range to about 10% of itself. Thus
    % (disregarding frame 1) most centroids' tension value is discernible and
    % only a few centroids saturate for low-K frames.
    % Range as in the "v" variable:
    % $$$   DDmu2v = [1 0 max(DDmu2(:))];
    % $$$   DDmu2v = [1 0 max(max(DDmu2(:,2:end)))];
    DDmu2v = [1 0 tens{2}*max(max(DDmu2(:,2:end)))];
    % Colormap for the tension
    % $$$   tmp = gray(256); tmp(:,2:3) = 0;		% "Redscale"
    tmp = jet(256);
    DDplotv = struct('type','img',...			% Plot type
        'cmap',tmp);				% Colormap for 'img*'
    % Fig. 12 (colorbar)
    DDmu2ticks = str2num(num2str(linspace(DDmu2v(2),DDmu2v(3),6),2));  % Ticks
    ENcolorbar(DDmu2ticks,12,11,DDplotv.cmap);
    set(12,'Name','Scale for fig. 11'); drawnow;
end

% Figure 100: objective function value.
% Figure 101: computation time.
if any(plotfig(100:102))
    Ks = []; Es = []; times = []; its = zeros(length(ENlist),1);
    Tens = [];
    for ENcounter = 2:length(ENlist)
        its(ENcounter) = its(ENcounter-1) + size(ENlist(ENcounter).stats.K,1);
        Ks = [Ks;ENlist(ENcounter).stats.K];
        Es = [Es;ENlist(ENcounter).stats.E];		% Objective function
        Tens = [Tens;ENlist(ENcounter).stats.tension_vec];
        times = [times;ENlist(ENcounter).stats.time];	% Computation time
    end
    its(1) = NaN;
    lKs = length(Ks);
    Vars = {Es,times};
    Varsn = length(Vars);
    Varsl = {'Objective function',...
        'Computation time'};
    VarsL = {'Objective function E',...
        {['Computation time in seconds (means: ' ...
        num2str(mean(times),'%0.3g ') ')'],ENlist(end).stats.cpu}};
    Varsax = zeros(Varsn,4);
    axm = min(Es(:)); axM = max(Es(:)); axr = axM - axm;
    Varsax(1,:) = [1 lKs axm axM] + 0.05*(lKs-1)*[-1 1 0 0] + 0.1*axr*[0 0 -1 1];
    axm = 0; axM = max(times(:)); axr = axM - axm;
    Varsax(2,:) = [1 lKs axm axM] + 0.05*(lKs-1)*[-1 1 0 0] + 0.1*axr*[0 0 0 1];
    Figs = 100; Figs = Figs:Figs+Varsn-1;
    Figss = [560 420];
    Figsp = repmat([1 sch+1-Figss(2)-wnt-wnb],Varsn,1);
    Figsp(:,1) = Figsp(:,1) + (0:Varsn-1)'*(Figss(1)+2*wnb);
    
    for i=1:Varsn
        if plotfig(Figs(i))
            figure(Figs(i)); clf;
            set(gcf,'Position',[Figsp(i,:) Figss],'Name',Varsl{i},...
                'Color','w','MenuBar','none','PaperPositionMode','auto');
            plot(Vars{i},'-'); axis(Varsax(i,:));
            XT = get(gca,'XTick');
            XT = XT(((floor(XT)-XT)==0) & (XT>0) & (XT<=lKs));
            if XT(1) > 1 XT = [1 XT]; end
            if XT(end) < lKs XT = [XT lKs]; end
            set(gca,'XTick',XT);
            xlabel('Iteration'); ylabel(VarsL{i},'VerticalAlignment','middle');
            text(XT,repmat(Varsax(i,4)+0.025*diff(Varsax(i,3:4)),size(XT)),...
                num2str(Ks(XT),3),'HorizontalAlignment','center','FontSize',6);
            text(mean(Varsax(i,1:2)),Varsax(i,4)+0.065*diff(Varsax(i,3:4)),'K');
            if Figs(i) == 100
                legend('Total','Fitness term','Tension term with beta','Location','best'); % -- updated by Wei May 31 2019
            else
                legend('Total','Fitness term','Tension term with beta','Location','best');
            end
            %legend('Total','Fitness term','Tension term',0);
        end
    end
    if plotfig(102)
        figure(102);
        plot(Tens);
        legend('Tens VFx', 'Tens VFy', 'Tens OD', 'Tens ORx', 'Tens ORy', 'Location','best');
		title('Tension terms without beta');
    end
end

% Animation loop
for ENcounter = whichones
    % Extract current net
    mu = ENlist(ENcounter).mu;
    thisK = ENlist(ENcounter).stats.K(end);
    thisE = ENlist(ENcounter).stats.E(end,:);
    thistime = ENlist(ENcounter).stats.time(end,:);
    Kstr = ['K = ' num2str(thisK) ' (frame ' num2str(ENcounter) ')'];
    % Augment mu with (theta,r) values for OR
    if isfield(id, 'OR')
        mu = [mu zeros(size(mu,1),2)];
        [mu(:,end-1), mu(:,end)] = cart2pol(mu(:,id.ORx),mu(:,id.ORy));
        mu(:,end-1) = mu(:,end-1) / 2;
    end
    % Plot current net
    if plotfig(1) && isfield(id, 'OD')
        set(0,'CurrentFigure',1); cla;
        tmp = get(1,'Position'); fg1(1:2) = tmp(1:2);
        myplot(G,bc,mu,'img',v(id.OD,:),T,Pi,fg1,ax1);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(2) && isfield(id, 'OD') && isfield(id, 'OR')
        set(0,'CurrentFigure',2); cla;
        tmp = get(2,'Position'); fg2(1:2) = tmp(1:2);
        myplot(G,bc,mu,ODplotv,v(id.OD,:),T,Pi,fg2,ax2);
        myplot(G,bc,mu,ORplotv,v(id.OR,:),T,Pi,fg2,ax2);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(3) && isfield(id, 'OR')
        set(0,'CurrentFigure',3); cla;
        tmp = get(3,'Position'); fg3(1:2) = tmp(1:2);
        myplot(G,bc,mu,'img_per',v(id.OR,:),T,Pi,fg3,ax3);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(4) && isfield(id, 'OR') && isfield(id, 'ORr')
        set(0,'CurrentFigure',4); cla;
        tmp = get(4,'Position'); fg4(1:2) = tmp(1:2);
        myplot(G,bc,mu,'img_per',v([id.OR, id.ORr],:),T,Pi,fg4,ax4);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(5) && isfield(id, 'VFx') && isfield(id, 'VFy')
        set(0,'CurrentFigure',5); cla;
        tmp = get(5,'Position'); fg5(1:2) = tmp(1:2);
        myplot(G,bc,mu,proj_cart_plotv,v([id.VFx, id.VFy],:),T,Pi,fg5,ax5);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(6) && isfield(id, 'ORx') && isfield(id, 'ORy')
        set(0,'CurrentFigure',6); cla;
        tmp = get(6,'Position'); fg6(1:2) = tmp(1:2);
        myplot(G,bc,mu,proj_polar_plotv,v([id.ORx, id.ORy],:),T,Pi,fg6,ax6);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(7) && isfield(id, 'ORr')
        set(0,'CurrentFigure',7); cla;
        tmp = get(7,'Position'); fg7(1:2) = tmp(1:2);
        myplot(G,bc,mu,'img',v(id.ORr,:),T,Pi,fg7,ax7);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(8) && isfield(id, 'VFx') && isfield(id, 'VFy') && isfield(id, 'OD')
        set(0,'CurrentFigure',8); cla;
        tmp = get(8,'Position'); fg8(1:2) = tmp(1:2);
        myplot(G,bc,mu,proj_cart_plotv,v([id.VFx,id.VFy,id.OD],:),T,Pi,fg8,ax8);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(9) && isfield(id, 'ORx') && isfield(id, 'ORy') && isfield(id, 'OD')
        set(0,'CurrentFigure',9); cla;
        tmp = get(9,'Position'); fg9(1:2) = tmp(1:2);
        myplot(G,bc,mu,proj_cart_plotv,v([id.ORx, id.ORy, id.OD],:),T,Pi,fg9,ax9);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(10) && isfield(id, 'VFx') && isfield(id, 'VFy') && isfield(id, 'ORr')
        set(0,'CurrentFigure',10); cla;
        tmp = get(10,'Position'); fg10(1:2) = tmp(1:2);
        myplot(G,bc,mu,proj_cart_plotv,v([id.VFx, id.VFy, id.ORr],:),T,Pi,fg10,ax10);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(11) && exist('tens','var')
        set(0,'CurrentFigure',11); cla;
        tmp = get(11,'Position'); fg11(1:2) = tmp(1:2);
        myplot(G,bc,DDmu2(:,ENcounter),DDplotv,DDmu2v,T,Pi,fg11,ax11);
        title(Kstr,'Visible','on'); drawnow;
    end
    if plotfig(100)
        set(0,'CurrentFigure',100);
        hold on; plot(its(ENcounter),thisE,'ro'); hold off; drawnow;
    end
    if plotfig(101)
        set(0,'CurrentFigure',101);
        hold on; plot(its(ENcounter),thistime,'ro'); hold off; drawnow;
    end
    % Other options
    if domovie
        for f=1:10
            if plotfig(f)
                ENmovie(f).movie(moviecounter) = getframe(f);
            end
        end
        moviecounter = moviecounter + 1;
    elseif strcmp(opt,'pause')
        pause
    end
end

switch domovie
    case 1
        for f=1:10
            if plotfig(f)
                movie2avi(ENmovie(f).movie,[opt(1:end-4) num2str(f)],...
                    'compression',compression,'fps',3);
            end
        end
    case 2
        for f=1:10
            if plotfig(f)
                mpgwrite(ENmovie(f).movie,[],[opt(1:end-4) num2str(f)]);
            end
        end
end

