% stats = ENV1stats2(G,bc,ENlist,v[,whichones,T,Pi,murange,tens,opt,figlist])
% Quantitative measurements of a collection of 2D generalised elastic nets
% for cortical map modelling.
%
% The following plots can be produced:
%
% * Figures 1-12 plot a view of the net for the current frame (they are
%   the same figures that ENV1replay2 produces).
%
% - Figure 1: ocular dominance map.
% - Figure 2: OD-OR contours.
% - Figure 3: orientation angle map.
% - Figure 4: orientation polar map.
% - Figure 5: retinotopy (VFx,VFy) in Cartesian coordinates.
% - Figure 6: orientation (ORx,ORy) in polar coordinates.
% - Figure 7: orientation selectivity (ORr) map.
% - Figure 8: retinotopy + OD (VFx,VFy,OD) in Cartesian coordinates.     [SLOW]
% - Figure 9: orientation + OD (ORx,ORy,OD) in polar coordinates.        [SLOW]
% - Figure 10: retinotopy + ORr (VFx,VFy,ORr) in Cartesian coordinates.  [SLOW]
% - Figure 11: tension (not including the beta/2 factor).
% - Figure 12: colorbar for fig. 11.
%
% * Figures 20-99 plot a relevant statistic (e.g. a histogram) for the
%   current frame.
%
% - Figure 20: unweighted histogram of VFx.
% - Figure 21: unweighted histogram of VFy.
% - Figure 22: unweighted histogram of OD.
% - Figure 23: unweighted histogram of ORt.
% - Figure 24: unweighted histogram of ORr.
%
% - Figure 30: weighted histogram of crossing angles between OD and OR
%              (interior points).
% - Figure 31: weighted histogram of crossing angles between OD and OR
%              (boundary points).
% - Figure 32: weighted histogram of crossing angles of OD with map
%              boundaries.
% - Figure 33: weighted histogram of crossing angles of OR with map
%              boundaries.
% - Figure 34: combination of figs. 30-33.
%
% - Figure 40: Fourier transform of OD.
% - Figure 41: Fourier transform of ORt.
%
% - Figure 50: pinwheels' locations wrt OD borders, number, density & sign.
% - Figure 51: unweighted histogram of the distance pinwheel-to-OD-border
%              (all signs).
% - Figure 52: unweighted histogram of the distance pinwheel-to-OD-border
%              (positive).
% - Figure 53: unweighted histogram of the distance pinwheel-to-OD-border
%              (negative).
% - Figure 54: unweighted histogram of the distance pinwheel-to-OD-border
%              for randomly distributed pinwheels (all signs); plus
%              combination of figs. 51-53.                               [SLOW]
% - Figure 55: unweighted histogram of the distance between closest pairs
%              of pinwheels (all signs/all signs).
% - Figure 56: unweighted histogram of the distance between closest pairs
%              of pinwheels (positive/positive).
% - Figure 57: unweighted histogram of the distance between closest pairs
%              of pinwheels (negative/negative).
% - Figure 58: unweighted histogram of the distance between closest pairs
%              of pinwheels (positive/negative).
% - Figure 59: unweighted histogram of the distance between closest pairs
%              of randomly distributed pinwheels (all signs/all signs); plus
%              combination of figs. 55-58.                               [SLOW]
% Important note: the axes of the histograms for figs. 51-54 (and 55-59) are
% given by the maximal distance. If fig. 54 (resp. 59) is not plotted, the
% axes will always be the same (and so will the histogram statistics KLu,
% etc.); but if it is plotted, then each run of ENV1stats2 for figs. 51-59
% will be slightly different (because figs. 54 and 59 generate random
% distances).
%
% - Figure 70: correlations between the gradients of VF and OD.
% - Figure 71: correlations between the gradients of VF and ORt.
% - Figure 72: correlations between ORr and the gradient of VF.
% - Figure 73: correlations between the gradients of OD and ORt.
% - Figure 74: correlations between ORr and the gradient of OD.
% - Figure 75: correlations between ORr and the gradient of ORt.
% - Figure 76: unweighted histogram of the product of moduli of OD and OR
%              gradients.
%
% - Figure 80: weighted histogram of gradient direction of VFx
%              (interior points).
% - Figure 81: ditto VFy.
% - Figure 82: ditto OD.
% - Figure 83: ditto ORt.
% - Figure 84: ditto ORr.
% - Figure 85: weighted histogram of gradient direction of VFx
%              (boundary points).
% - Figure 86: ditto VFy.
% - Figure 87: ditto OD.
% - Figure 88: ditto ORt.
% - Figure 89: ditto ORr.
% - Figure 90: combination of figs. 80-84.
% - Figure 91: combination of figs. 85-89.
%
% - Figure 95: gradient of VF and contours of OD and OR.
% - Figure 96: gradient of OD and contours of OD and OR.
% - Figure 97: gradient of ORt and contours of OD and OR.
%
% * Figures 100-176 are time-course figures, i.e., they plot the relevant
%   statistic (e.g. a histogram mean) as a function of the frame number.
%
% - Figure 100: objective function value.
% - Figure 101: computation time.
%
% - Figure 120: histogram statistics for fig. 20 (KLu,L2u,m,s,g1).
% - Figure 121: ditto fig. 21.
% - Figure 122: ditto fig. 22.
% - Figure 123: ditto fig. 23.
% - Figure 124: ditto fig. 24.
%
% - Figure 130: histogram statistics for fig. 30 (KLu,L2u,m,s,g1).
% - Figure 131: ditto fig. 31.
% - Figure 132: ditto fig. 32.
% - Figure 133: ditto fig. 33.
%
% - Figure 140: Fourier spectrum statistics for fig. 40
%               (lambdaM,alphaM,lambdam).
% - Figure 141: ditto fig. 41.
% - Figure 142: combination of figs. 140-141.
%
% - Figure 150: pinwheel statistics for fig. 50 (numbers & density of
%               pinwheels for both signs).
% - Figure 151: histogram statistics for fig. 51 (KLu,L2u,m,s,g1).
% - Figure 152: ditto fig. 52.
% - Figure 153: ditto fig. 53.
% - Figure 154: ditto fig. 54.                                           [SLOW]
% - Figure 155: ditto fig. 55.
% - Figure 156: ditto fig. 56.
% - Figure 157: ditto fig. 57.
% - Figure 158: ditto fig. 58.
% - Figure 159: ditto fig. 59.                                           [SLOW]
% - Figure 160: pinwheel statistics for pinwheels-on-OD-borders (numbers
%               & density of pinwheels for both signs).
%
% - Figure 170: correlation statistics for figs. 70-75 (C).
% - Figure 171: correlation statistics for figs. 70-75 (c).
% - Figure 176: histogram statistics for fig. 76 (KLu,L2u,m,s,g1).
%
% - Figure 180: histogram statistics for fig. 80 (KLu,L2u,m,s,g1).
% - Figure 181: ditto fig. 81.
% - Figure 182: ditto fig. 82.
% - Figure 183: ditto fig. 83.
% - Figure 184: ditto fig. 84.
% - Figure 185: ditto fig. 85.
% - Figure 186: ditto fig. 86.
% - Figure 187: ditto fig. 87.
% - Figure 188: ditto fig. 88.
% - Figure 189: ditto fig. 89.
%
% In: (see ENV1replay2 for arguments not listed here)
%   opt: '' (to plot to the screen), 'pause' (to plot to the screen and
%      wait for a keystroke between successive plots), 'DDD.EXT' (to produce
%      output of type EXT under directory DDD). Default: ''.
%      EXT can be any format recognised by Matlab (epsc, png, tiff...);
%      see Matlab's help for the "print" command.
%      The output (EPS files, etc.), if any, is left in files
%      "grf/DDD/frameNNNN/figFFF.EXT" where NNNN indicates the frame number
%      and FFF the figure number.
%   figlist: list of numbers corresponding to the figures to plot. Default:
%      the time-course figures. To plot all figures use 1:1000.
% Out:
%   stats: structure with summary statistics for each of the time-course
%      figures as given by figures 100-176 (useful to produce tables and
%      compare with other simulations). It contains the following fields:
%      . 'whichones': list of indices in ENlist, indicating for which
%        frames the statistics were computed.
%      . 'Ks': ?x1 list of K values for all frames in ENlist (each frame
%        can contain several values of K, depending on how many iterations
%        it contains).
%      . 'its': Fx1 list containing the cumulative number of iterations for
%        each frame (F = length(ENlist)).
%      . 'fig': structure array containing the statistics for each figure.
%        For example, if W is the last frame in "whichones", then
%        stats.fig(140).lambdam is a 1xW cell array containing the OD
%        wavelengths ("mean" method) for the "whichones" frames (elements
%        not in "whichones" are empty). Note: the statistics are computed
%        for the frames given in "whichones" and the figures given in
%        "figlist" only. For more details about each statistic, look for the
%        appropriate figure in the code below.
%      'whichones', 'Ks' and 'its' are necessary to plot the right
%      coordinates along the horizontal axis.
%
% Notes:
%
% - The arguments of ENV1stats2 have the same meaning as for ENV1replay2,
%   with "whichones" selecting what frames to plot, "figlist" selecting what
%   figures to plot and "opt" indicating what type of output to produce
%   (screen with or without pause, or EPS/GIF/etc. figures).
%
% - If whichones contains a single element, no time-course figures are
%   plotted, but the summary statistics are still returned in "stats". This
%   is useful to gather statistics of different simulation files at a single
%   frame to then draw curves with errorbars.
%
% - See other notes as in ENV1replay2.
%
% - Regarding the "print" formats:
%   . The most useful formats should be png (for screen, web page) and eps
%     and epsc (for documents, in BW and colour, respectively).
%   . The settings given to "print" below should work most times; if not,
%     do it manually or edit ENV1stats2 as desired. The same if one is not
%     happy with the title, labels or other figure properties.
%   . For eps*, the setting '-loose' produces a bounding box to match the
%     screen appearance of the figure, rather than a tight one. This is
%     useful to force all figures to have the same dimensions and be able
%     to align them.
%   . For png, the setting '-r150' produces an image file with the same
%     dimensions as in the screen. For other computers this may need to be
%     adjusted.
%   . The "printed" figures sometimes have slight differences with the
%     screen ones; I think this is a problem of Matlab's driver. Specifically:
%     myplot figures lose the left & top lines of the box in png; fig. 52
%     shows a spurious blue line in epsc; in fig. 40-41 the image colours
%     are slightly different; titles and labels may be slightly offset.
%   . Matlab has a "saveas" command to produce graphic files out of a figure
%     but its functionality is more limited than that of "print".
%
% To do:
%
% - The vertical axis limits and ticks in some time-course figures (forced
%   to be round numbers) can occasionally fail and give an error "Bad value
%   for axes property: 'YLim', Values must be increasing and non-NaN" or
%   "Values must be monotonically increasing". This occurs when the Y
%   variable is almost constant.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.
%
% See also ENV1replay2, ENV1driver2, ENV1ODbord, ENV1pinw, ENgrad,
% ENxangle, myfft2, ENcorr.

% Copyright (c) 2002 by Miguel A. Carreira-Perpinan

function stats = ...
    myV1stats(G,bc,ENlist,v,whichones,T,Pi,murange,id,tens,opt,figlist,statsOnly)
if nargin < 13
    statsOnly = false;
end

L = length(G);                  % Net dimensionality
if L == 1 G = [G 1]; end        % So that both L=1 and L=2 work
[M,D] = size(ENlist(1).mu);

% Argument defaults
if ~exist('whichones','var') | isempty(whichones)
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

if ~exist('T','var') | isempty(T) T = repmat(NaN,1,D); end
% Augment T with (theta,r) values for OR
if isfield(id, 'OR')
    T = [T zeros(size(T,1),2)];
    [T(:,end-1) T(:,end)] = cart2pol(T(:,id.ORx),T(:,id.ORy));
    T(:,end-1) = T(:,end-1) / 2;
end

if ~exist('Pi','var') || isempty(Pi)
    Pi = ones(1,M)/M;
    zPi = [];
else
    Pi = Pi/sum(Pi);	% Make sure the mixing proportions are normalised
    zPi = find(Pi==0);
end

disp(['ratio of L/R = ', num2str(sum(ENlist(end).mu(logical(Pi),id.OD) < 0)/sum(ENlist(end).mu(logical(Pi),id.OD) > 0))]);

% Ranges for all variables
if ~exist('murange','var') | isempty(murange)
    all = cat(1,ENlist.mu);
    if isfield(id, 'OR')
        all = [all zeros(size(all,1),2)];
        [all(:,end-1) all(:,end)] = cart2pol(all(:,id.ORx),all(:,id.ORy));
        all(:,end-1) = all(:,end-1) / 2;
    end
    all = [all;T];
else
    all = cat(1,T,murange);
end
all = [min(all);max(all)];

if ~exist('tens','var') tens = []; end;

ENdir = [];
if ~exist('opt','var') || isempty(opt)
    opt = '';
else
    tmp = findstr('.',opt);
    if ~isempty(tmp)		% Print figures
        ENdir = opt(1:tmp(end)-1); EXT = lower(opt(tmp(end)+1:end));
    end
end

if ~exist('figlist','var')
    if isempty(figlist) && ~statsOnly
        figlist = [1:4 7 100:101 120:124 130:133 140:142 150:160 170:171 176 180:189];
    else
        figlist = [];
    end
end
% plotfig(i) = 1 means plot figure(i).
plotfig = logical(zeros(1,1000)); plotfig(figlist) = 1;

% Output argument
stats = struct('whichones',whichones,'Ks',[],'its',[],'fig',[]);

% --------------------------------------------------------------------------
% Some general variables

% For nonperiodic boundary conditions, in some of the calculations below
% we deal separately with the boundary points (frame of width in pixels
% "B_width" >= 0 around the cortex) than with the interior points.
if strcmp(bc,'periodic')
    B_width = 0;
    I_ind = (1:M)';
    B_ind = [];
else
    test_boundary = false;
    B_width = 5;
    [B_ind, I_ind, I1_ind, B1_ind, B1_ang] = set_bc(reshape(logical(Pi),G(1),G(2)), B_width, test_boundary);
    
    %     [tmp1,tmp2] = ndgrid(B_width+1:G(1)-B_width,B_width+1:G(2)-B_width);
    %     I_ind = sub2ind(G,tmp1(:),tmp2(:));
    %     tmp = ones(G); tmp(I_ind) = 0;
    %     B_ind = find(tmp);
    % $$$   % You can visually check the boundary/interior regions like this:
    if test_boundary
        tmp = ones(G); tmp(I_ind) = 0;
        figure(910); clf; myplot(G,bc,tmp(:),'img',[1 0 1]);
        tmp = ones(G); tmp(B_ind) = 0;
        figure(911); clf; myplot(G,bc,tmp(:),'img',[1 0 1]);
        tmp = ones(G); tmp(B1_ind) = 0;
        figure(912); clf; myplot(G,bc,tmp(:),'img',[1 0 1]);
    end
end

% We also compute the crossing angles of OD and OR with the map boundary (of
% width 1 pixel).
%B1_ind = [(2:G(1)-1)' ones(G(1)-2,1);(2:G(1)-1)' repmat(G(2),G(1)-2,1);...
%     ones(G(2)-2,1) (2:G(2)-1)';repmat(G(1),G(2)-2,1) (2:G(2)-1)'];
%B1_ind = sub2ind(G,B1_ind(:,1),B1_ind(:,2));
% WRONG!!! row G(1) is y, column G(2) is x
%B1_ang = [repmat(0,2*(G(1)-2),1);repmat(pi/2,2*(G(2)-2),1)];

% Number of bins for the histograms of crossing angles, etc.
nbins = 10;

% Number of trials for random histogram generation
NTrials = 50;

% Maximum distance between a pinwheel and an OD border for the pinwheel to
% be considered as on the OD border, in pixels. We need a slightly large
% threshold because the OD borders are represented by a discrete collection
% of points, not by a continuous curve, and so even if a pinwheel is right
% on an OD border it may lie at some distance from the nearest OD border
% point.
OD_pinw_th = 1;

% Contour lines properties for OD (see myplot)
ODplotv = struct('type','contour',...             % Plot type
    'cmap',zeros(256,3),...          % Colormap for 'img*'
    'line',struct('lsty','-',...     % LineStyle
    'lcol','k',...     % LineColor
    'lwid',3,...       % LineWidth
    'msty','none',...  % MarkerStyle
    'msiz',1,...       % MarkerSize
    'mecol','none',... % MarkerEdgeColor
    'mfcol','none',... % MarkerFaceColor
    'num',[1 1]*mean(v(id.OD,2:3))));% Single line
%                               'num',1));         % No. lines for 'contour*'
% Ditto OR
ORplotv = ODplotv;
ORplotv.type = 'contour_per';
ORplotv.line.lwid = 1;
ORplotv.line.num = linspace(v(id.OR,2),v(id.OR,3),17);
ORplotv.line.lcol = 'k';
% Inverted greyscale colormap for Fourier domain plots (figs. 40-41)
tmp = gray(256);
FTplotv = struct('type','img',...			% Plot type
    'cmap',tmp(end:-1:1,:));		% Colormap for 'img*'
% Special colormap for gradient plots (figs. 95-97)
tmp = jet(256);
GRplotv = struct('type','img',...			% Plot type
    'cmap',tmp(32:end-31,:));		% Colormap for 'img*'

[scw,sch,fgt,fgb,wnt,wnb] = ENscrinfo;			% Screen geometry
Figl = floor(min(scw,sch)/4);				% Largest dimension
% --------------------------------------------------------------------------

if ~isempty(ENdir)
    [tmp,tmp] = mkdir(ENdir);
end

% "firstone" is used to set to create and the figure position only the
% first time in the ENcounter loop.
firstone = logical(1);
for ENcounter = whichones
    
    if ~isempty(ENdir)
        frame = [ENdir,'/frame' num2str(ENcounter,'%04d')];
    end
    
    % Extract current net
    mu = ENlist(ENcounter).mu;
    mu(zPi,:) = NaN;			% Disabled centroids aren't plotted
    Kstr = ['K = ' num2str(ENlist(ENcounter).stats.K(end)) ...
        ' (frame ' num2str(ENcounter) ')'];
    % Augment mu with (theta,r) values for OR
    if isfield(id, 'OR')
        mu = [mu zeros(size(mu,1),2)];
        [mu(:,end-1) mu(:,end)] = cart2pol(mu(:,id.ORx),mu(:,id.ORy));
        mu(:,end-1) = mu(:,end-1) / 2;
    end
    %
    % Now, the columns of each of mu correspond to the following:
    % 1. VFx (visual field x)
    % 2. VFy (visual field y)
    % 3. OD  (ocular dominance)
    % 4. ORx (orientation x)
    % 5. ORy (orientation y)
    % 6. ORt (orientation angle)
    % 7. ORr (orientation modulus, or selectivity)
    
    % ------------------------------------------------------------------------
    % Compute gradients
    % [gx,gy,gt,gr] = ENgrad(G,mu,v(id.OR,:));
    [gx,gy,gt,gr] = myGrad(G,mu,B1_ind,I1_ind,logical(Pi),v(id.OR,:));
    % Augment gr with the visual field gradient gVF, obtained simply as the
    % sum of the gradient moduli of VFx and VFy:
    gr = [gr sum(gr(:,[id.VFx id.VFy]),2)];
    % Another possibility would be to obtain gVF as sqrt(gVFx² + gVFy²):
    % $$$   gr = [gr sqrt(sum(gr(:,[1 2]).^2,2))];
    % which doesn't make much difference in practice.
    % What is wrong is to obtain gVF as the vectorial sum for the gradients
    % of VFx and VFy):
    % $$$   gx = [gx sum(gx(:,[1 2]),2)]; gy = [gy sum(gy(:,[1 2]),2)];
    % $$$   [tmp1 tmp2] = cart2pol(gx(:,8),gy(:,8));
    % $$$   gt = [gt tmp1]; gr = [gr tmp2];
    % because this leads to cancellation of gVF if gVFx and gVFy point in
    % opposite directions. On the average, the difference with the two
    % options isn't large though.
    %
    % Now, the columns of each of gx, gy, gt, gr correspond to the following:
    % 1. gVFx (gradient of the visual field x)
    % 2. gVFy (gradient of the visual field y)
    % 3. gOD  (gradient of the ocular dominance)
    % 4. gORx (gradient of the orientation x)
    % 5. gORy (gradient of the orientation y)
    % 6. gORt (gradient of the orientation angle)
    % 7. gORr (gradient of the orientation modulus, or selectivity)
    % And gr has an 8th column:
    % 8. gVF  (gradient of the visual field)
    % And for a given of those columns:
    % gx = x-coordinate of the gradient | Cartesian
    % gy = y-coordinate of the gradient | coordinates
    % gt = polar angle of the gradient  | Polar
    % gr = modulus of the gradient      | coordinates
    % ------------------------------------------------------------------------
    
    
    % --- Figures 1-12,100-101: plot current net
    myV1replay(G,bc,ENlist,v,ENcounter,T(:,1:5),Pi,murange,id,tens,[],figlist);

    if ~isempty(ENdir)
        Figs = [1:12, 100:102];		% Energy and cpu time get printed at end only
        for i=find(plotfig(Figs))
            print(i,'-loose','-r150',['-d' EXT],sprintf('%s-fig%03d.%s',frame,i,EXT));
        end
    end
    if plotfig(100)
        stats.fig(100).K{ENcounter} = ENlist(ENcounter).stats.K;
        stats.fig(100).E{ENcounter} = ENlist(ENcounter).stats.E;
    end
    if plotfig(101)
        stats.fig(101).K{ENcounter} = ENlist(ENcounter).stats.K;
        stats.fig(101).time{ENcounter} = ENlist(ENcounter).stats.time;
    end
    
    
    % --- Figures 20-24: unweighted histograms of the maps VFx,VFy,OD,ORt,ORr
    % Note: for the standard EN, the histograms are strongly quantised to the
    % training set values.
    Varsi = [1 2 3 6 7];
    Varsn = length(Varsi);
    Vars = mu(logical(Pi),Varsi);				% VFx,VFy,OD,ORt,ORr
    VarsL = {'Visual field X',...
        'Visual field Y',...
        'Ocular dominance',...
        'Orientation angle',...
        'Orientation selectivity'};
    Varsl = {'VFx','VFy','OD','ORt','ORr'};
    Varsax = [min(Vars,[],1);max(Vars,[],1)];
    Varsax = Varsax + [-1;1]*diff(Varsax,1,1)/10;
    Figs = 20; Figs = Figs:Figs+Varsn-1;
    Figsp = [(linspace(1,scw/2,Varsn))' repmat(sch/2,Varsn,1)];
    Figss = [560 420]*2;
    for i=1:Varsn
        % Compute histogram and statistics if required here or elsewhere
        if any(plotfig([Figs(i) Figs(i)+100]))
            [h,KLu,L2u,m,s,g1] = ENhist(Vars(:,i),[],50);
            stats.fig(Figs(i)+100).KLu{ENcounter} = KLu;
            stats.fig(Figs(i)+100).L2u{ENcounter} = L2u;
            stats.fig(Figs(i)+100).m{ENcounter} = m;
            stats.fig(Figs(i)+100).s{ENcounter} = s;
            stats.fig(Figs(i)+100).g1{ENcounter} = g1;
        end
        % Plot this figure
        if plotfig(Figs(i))
            if firstone
                if ~ishandle(Figs(i))
                    figure(Figs(i));
                end
                set(Figs(i),'Position',[Figsp(i,:) Figss],...
                    'Name',['Histogram of ' Varsl{i}],...
                    'Color','w','PaperPositionMode','auto');
            end
            set(0,'CurrentFigure',Figs(i));
            bar(h(:,2),h(:,1),1);
            tmp = get(gca,'YLim');
            hold on;					% Nominal domain
            plot([1 1]*v(Varsi(i),2),tmp,'r--');
            plot([1 1]*v(Varsi(i),3),tmp,'r--');
            hold off;
            set(gca,'XLim',[min(h(1,3),v(Varsi(i),2)) max(h(end,4),v(Varsi(i),3))]);
            xlabel(VarsL{i});
            title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
                ';   \mu=' num2str(m,3) ';   \sigma=' num2str(s,3) ...
                ';   \gamma_1=' num2str(g1,3)],'Visible','on');
            drawnow;
            if ~isempty(ENdir)
                print(Figs(i),'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
            end
        end
    end
    
    % --- Figures 30-31: crossing angles between OD and ORt in [0,90] degrees
    % Plot the histogram of crossing angles and display statistics of it
    Figss = [560 420];
    Figsp = [linspace(1,1+Figss(1),2)' repmat(sch+1-Figss(2)-wnt-wnb,2,1)];
    xangh = zeros(nbins,4);			% For fig. 34
    
    % Interior points
    Figs = 30;
    % Compute histogram and statistics if required here or elsewhere
    if any(plotfig([Figs Figs+100 34])) || statsOnly
        xangI_OD_OR = ENxangle(gt(I_ind,id.OD),gt(I_ind,id.OR))*180/pi;
        % Histogram weighting angles according to their combined moduli of OD
        % and ORt gradients:
        [h,KLu,L2u,m,s,g1] = ...
            ENhist(xangI_OD_OR,gr(I_ind,id.OD).*gr(I_ind,id.OR),nbins,[0 90]);
        xangh(:,1) = h(:,1);
        stats.fig(Figs+100).KLu{ENcounter} = KLu;
        stats.fig(Figs+100).L2u{ENcounter} = L2u;
        stats.fig(Figs+100).m{ENcounter} = m;
        stats.fig(Figs+100).s{ENcounter} = s;
        stats.fig(Figs+100).g1{ENcounter} = g1;
        [amp, stats.OD_OR_I_angMode] = max(h(:,1));
        stats.OD_OR_I_ang_overMean = amp/mean(h(:,1));
        stats.OD_OR_I_angMode = h(stats.OD_OR_I_angMode,2);
    end
    % Plot this figure
    if plotfig(Figs)
        if firstone
            if ~ishandle(Figs)
                figure(Figs);
            end
            set(Figs,'Position',[Figsp(1,:) Figss],...
                'Name','Histogram of OD/ORt crossing angles (interior points)',...
                'Color','w','PaperPositionMode','auto');
        end
        set(0,'CurrentFigure',Figs);
        plot([0 90],[1 1]/90,'r--');          % Uniform distribution
        hold on; bar(h(:,2),h(:,1),1); hold off;
        set(gca,'XLim',[0 90]);
        xlabel('Crossing angle in degrees (\circ)');
        title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
            ';   \mu=' num2str(m,3) '\circ;   \sigma=' num2str(s,3) ...
            '\circ;   \gamma_1=' num2str(g1,3)],'Visible','on');
        drawnow;
        if ~isempty(ENdir)
            print(Figs,'-loose','-r150',['-d' EXT],sprintf('%s-fig%03d.%s',frame,Figs,EXT));
        end
    end
    
    % Boundary points
    Figs = 31;
    % Compute histogram and statistics if required here or elsewhere
    if any(plotfig([Figs Figs+100 34])) || statsOnly
        if B_width > 0
            xangB_OD_OR = ENxangle(gt(B_ind,id.OD),gt(B_ind,id.OR))*180/pi;
            [h,KLu,L2u,m,s,g1] = ...
                ENhist(xangB_OD_OR,gr(B_ind,id.OD).*gr(B_ind,id.OR),nbins,[0 90]);
        else
            [h,KLu,L2u,m,s,g1] = deal(NaN);
        end
        xangh(:,2) = h(:,1);
        stats.fig(Figs+100).KLu{ENcounter} = KLu;
        stats.fig(Figs+100).L2u{ENcounter} = L2u;
        stats.fig(Figs+100).m{ENcounter} = m;
        stats.fig(Figs+100).s{ENcounter} = s;
        stats.fig(Figs+100).g1{ENcounter} = g1;
        [amp, stats.OD_OR_B_angMode] = max(h(:,1));
        stats.OD_OR_B_ang_overMean = amp/mean(h(:,1));
        stats.OD_OR_B_angMode = h(stats.OD_OR_B_angMode,2);
    end
    % Plot this figure
    if plotfig(Figs)
        if firstone
            if ~ishandle(Figs)
                figure(Figs);
            end
            set(Figs,'Position',[Figsp(2,:) Figss],...
                'Name',...
                'Histogram of OD/ORt crossing angles (boundary points)',...
                'Color','w','PaperPositionMode','auto');
        end
        set(0,'CurrentFigure',Figs);
        if B_width > 0
            plot([0 90],[1 1]/90,'r--');        % Uniform distribution
            hold on; bar(h(:,2),h(:,1),1); hold off;
            set(gca,'XLim',[0 90]);
            xlabel('Crossing angle in degrees (\circ)');
        else
            axis off; cla;
            text(0.5,0.5,'Not available: the boundary width is zero',...
                'HorizontalAlignment','center');
        end
        title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
            ';   \mu=' num2str(m,3) '\circ;   \sigma=' num2str(s,3) ...
            '\circ;   \gamma_1=' num2str(g1,3)],'Visible','on');
        drawnow;
        if ~isempty(ENdir)
            print(Figs,'-loose','-r150',['-d' EXT],sprintf('%s-fig%03d.%s',frame,Figs,EXT));
        end
    end
    
    % $$$ % Note that there are three ways of computing the histograms:
    % $$$ % 1. Weighted: each angle contributes to the histogram according to
    % $$$ %    its combined moduli of OD and ORt gradients, so that angles where
    % $$$ %    either OD or ORt are nearly constant are disregarded:
    % $$$ [h,KLu,L2u,m,s,g1] = ...
    % $$$     ENhist(xangI_OD_OR,gr(I_ind,3).*gr(I_ind,6),nbins,[0 90]);
    % $$$ % 2. Unweighted: angles are equally valid anywhere in the cortex:
    % $$$ [h,KLu,L2u,m,s,g1] = ENhist(xangI_OD_OR,[],nbins,[0 90]);
    % $$$ % 3. Thresholded: angles are valid only in areas with gradients
    % $$$ %    exceeding a given threshold; needs to set thresholds OD_gr_th
    % $$$ %    and ORt_gr_th, typically to the first sharp elbow of the OD
    % $$$ %    and ORt gradient histograms in figure 999 (which could be done
    % $$$ %    automatically):
    % $$$ set(figure(900),'Position',[10 10 300 700]);
    % $$$ subplot(2,1,1); hist(gr(:,3),100); xlabel('OD gradient modulus');
    % $$$ subplot(2,1,2); hist(gr(:,6),100); xlabel('ORt gradient modulus');
    % $$$ OD_gr_th = 0.01; ORt_gr_th = 0.05; % These values usually work well
    % $$$ [h,KLu,L2u,m,s,g1] = ...
    % $$$     ENhist(xangI_OD_OR,...
    % $$$ 	   (gr(I_ind,3)>=OD_gr_th).*(gr(I_ind,6)>=ORt_gr_th),nbins,[0 90]);
    % $$$ % You can visually check the area above the threshold like this:
    % $$$ figure(901); cla;
    % $$$ myplot(G,bc,gr(:,3)>=OD_gr_th,'img',[1 0 1],[],Pi,[1 1 Figl]);
    % $$$ figure(902); cla;
    % $$$ myplot(G,bc,gr(:,6)>=ORt_gr_th,'img',[1 0 1],[],Pi,[1 1 Figl]);
    % $$$ figure(903); cla;
    % $$$ myplot(G,bc,(gr(:,3)>=OD_gr_th).*(gr(:,6)>=ORt_gr_th),...
    % $$$        'img',[1 0 1],[],Pi,[1 1 Figl]);
    
    
    % --- Figures 32-33: crossing angles of OD and ORt with map boundaries
    %     in [0,90] degrees
    Varsi = [3 6];
    Varsn = length(Varsi);
    Vars = gt(B1_ind,Varsi);			% OD,ORt
    Varsl = {'OD','ORt'};
    Figs = 32; Figs = Figs:Figs+Varsn-1;
    Figss = [560 420];
    Figsp = [linspace(1,1+Figss(1)+2*wnb,2)' ...
        repmat(sch+1-2*(Figss(2)+wnt+wnb),2,1)];
    for i=1:Varsn
        % Compute histogram and statistics if required here or elsewhere
        if any(plotfig([Figs(i) Figs(i)+100 34])) || statsOnly
            % Histogram weighting angles according to their moduli of OD gradient:
            [h,KLu,L2u,m,s,g1] = ...
                ENhist(ENxangle(Vars(:,i),B1_ang)*180/pi,...	% Map against boundary
                gr(B1_ind,Varsi(i)),nbins,[0 90]);
            xangh(:,i+2) = h(:,1);
            stats.fig(Figs(i)+100).KLu{ENcounter} = KLu;
            stats.fig(Figs(i)+100).L2u{ENcounter} = L2u;
            stats.fig(Figs(i)+100).m{ENcounter} = m;
            stats.fig(Figs(i)+100).s{ENcounter} = s;
            stats.fig(Figs(i)+100).g1{ENcounter} = g1;
            if i == 1
                [amp, stats.OD_B_angMode] = max(h(:,1));
                stats.OD_B_ang_overMean = amp/mean(h(:,1));
                stats.OD_B_angMode = h(stats.OD_B_angMode,2);
            else
                [amp, stats.OR_B_angMode] = max(h(:,1));
                stats.OR_B_ang_overMean = amp/mean(h(:,1));
                stats.OR_B_angMode = h(stats.OR_B_angMode,2);
            end
        end
        if plotfig(Figs(i))
            if firstone
                if ~ishandle(Figs(i))
                    figure(Figs(i));
                end
                set(Figs(i),'Position',[Figsp(i,:) Figss],...
                    'Name',['Histogram of ' Varsl{i} ...
                    '/boundary crossing angles'],...
                    'Color','w','PaperPositionMode','auto');
            end
            set(0,'CurrentFigure',Figs(i));
            % Plot the histogram of crossing angles and display statistics of it:
            plot([0 90],[1 1]/90,'r--');		% Uniform distribution
            hold on; bar(h(:,2),h(:,1),1); hold off;
            set(gca,'XLim',[0 90]);
            xlabel('Crossing angle in degrees (\circ)');
            title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
                ';   \mu=' num2str(m,3) '\circ;   \sigma=' num2str(s,3) ...
                '\circ;   \gamma_1=' num2str(g1,3)],'Visible','on');
            drawnow;
            if ~isempty(ENdir)
                print(Figs(i),'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
            end
        end
    end
    
    
    % --- Figure 34: combination of figs. 30-33.
    % Uses Figsp and h from the previous figure.
    Figs = 34;
    Figss = [560 420];
    Figsp = Figsp(1,:) + Figss/2;
    if plotfig(Figs)
        if firstone
            if ~ishandle(Figs)
                figure(Figs);
            end
            set(Figs,'Position',[Figsp Figss],...
                'Name',['Combined histograms of crossing angles'],...
                'Color','w','PaperPositionMode','auto');
        end
        set(0,'CurrentFigure',Figs);
        plot([0 90],[1 1]/90,'r--');		% Uniform distribution
        hold on; H = plot(h(:,2),xangh); hold off;
        set(gca,'XLim',[0 90],'YLim',[0 1.1*max(xangh(:))]);
        xlabel('Crossing angle in degrees (\circ)');
        set(H,'LineWidth',2);
        legend(H,'OD/ORt (interior points)','OD/ORt (boundary points)',...
            'OD/boundary','ORt/boundary','Location','best');
        drawnow;
        if ~isempty(ENdir)
            print(Figs,'-loose','-r150',['-d' EXT],...
                sprintf('%s-fig%03d.%s',frame,Figs,EXT));
        end
    end
    
    
    % --- Figures 40-41: Fourier transforms
    % We use the Fourier transform to obtain, for each of the OD and OR maps,
    % the dominant wavelength l in pixels of the stripes (defined as
    % orthogonal to the stripes) and the angle t that these stripes make with
    % the vertical, defined so that (when plotted with myplot):
    %   t =       -pi/2   -pi/4     0     pi/4    pi/2
    %   stripes =   -       /       |       \       -
    %
    % Should I do separate FFTs for interior and boundary? Too messy.
    %
    % Note that, for orientation, the right map to use is ORt = mu(:,6):
    % - ORx and ORy can't be used because, even though they have the right
    %   angular relation with ORt, the modulus ORr varies along the cortex.
    %   Thus ORx = r.sin(2 ORt) and ORy = r.cos(2 ORt) but r is not constant.
    % - ORt, sin(ORt) and cos(ORt) don't have the same Fourier spectrum (see
    %   myfft2); usually ORt and sin(ORt) are similar, with cos(ORt) being
    %   also similar but rotated +pi/2.
    Varsi = [id.OD id.OR];
    Varsn = length(Varsi);
    Vars = mu(:,Varsi);						% OD,ORt
    Varsl = {'OD','ORt'};
    Figs = 40; Figs = Figs:Figs+Varsn-1;
    Figsp = [repmat(scw/2,Varsn,1) (linspace(1,50+sch/4,Varsn))'];
    Figss = Figl;
    for i=1:Varsn
        % Compute DFT and statistics if required here or elsewhere
        if any(plotfig([Figs(i) Figs(i)+100 142 50:59 150:160 70:75 170:171])) || statsOnly
            [tmpfft,tmpkM,tmplM,tmptM,tmpkm,tmplm] = myfft2(G,Vars(:,i),Pi);
            switch i
                case 1
                    [ODfft,ODkM,ODlM,ODtM,ODkm,ODlm] = ...
                        deal(tmpfft,tmpkM,tmplM,tmptM,tmpkm,tmplm);		% OD
                case 2
                    [ORfft,ORkM,ORlM,ORtM,ORkm,ORlm] = ...
                        deal(tmpfft,tmpkM,tmplM,tmptM,tmpkm,tmplm);		% ORt
            end
            stats.fig(Figs(i)+100).lambdaM{ENcounter} = tmplM;
            stats.fig(Figs(i)+100).alphaM{ENcounter} = tmptM*180/pi;
            stats.fig(Figs(i)+100).lambdam{ENcounter} = tmplm;
        end
        % Plot this figure
        if plotfig(Figs(i))
            if firstone
                if ~ishandle(Figs(i))
                    figure(Figs(i));
                end
                set(Figs(i),'Position',[Figsp(i,:) 10 10],...
                    'Name',[Varsl{i} ' map Fourier spectrum']);
            end
            % $$$       % Plot square-root power instead of the power to see
            % $$$       % low-power values better
            % $$$       tmpfft = sqrt(tmpfft);
            set(0,'CurrentFigure',Figs(i)); cla;
            tmp = get(Figs(i),'Position'); Figsp(i,:) = tmp(1:2);
            % $$$       % Direct colormap (white = high)
            % $$$       myplot(G,bc,tmpfft,'img',[1 0 max(tmpfft)],[],Pi,[Figsp(i,:) Figss]);
            % Inverted colormap to see low-power values better
            myplot(G,bc,tmpfft,FTplotv,[1 0 max(tmpfft)],[],Pi,[Figsp(i,:) Figss]);
            title(['k = (' num2str(tmpkM(1),2) ',' num2str(tmpkM(2),2) ...
                ');  \lambda_M = ' num2str(tmplM,3) ...
                ' pixels;  \alpha_M = ' num2str(tmptM*180/pi,3) ...
                '\circ;  \lambda_m = ' num2str(tmplm,3) ' pixels'], ...
                'FontSize',8,'Visible','on');
            drawnow;
            if ~isempty(ENdir)
                print(Figs(i),'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
            end
        end
    end
    
    
    % --- Figure 50: orientation pinwheels
    
    % Compute pinwheel locations and statistics if required here or elsewhere
    if any(plotfig([50:59 150:160 70:75 170:171])) || statsOnly
        % Locate all the pinwheels:
        [ORpinw ORwindno] = ENV1pinw(G,[mu(:,[id.OR, id.ORr]) gr(:,id.OR)],v([id.OR, id.ORr],:));
        ORpinwp = ORpinw(ORwindno > 0,:);
        ORpinwn = ORpinw(ORwindno < 0,:);
        npinw = size(ORpinw,1)
        stats.npinw = npinw;
        npinwp = size(ORpinwp,1);
        npinwn = size(ORpinwn,1);
        
        % Number and density of pinwheels, stored in the 3x7 array pinw as
        % follows:
        % pinw(i,:): all, positive and negative pinwheels (i = 1..3, resp.).
        % pinw(:,j):
        %   j = 1: number of pinwheels
        %   j = 2: pinwheel density per square pixel
        %   j = 3: pinwheel density per OR module (mode)
        %   j = 4: pinwheel density per OD module (mode)
        %   j = 5: pinwheel density per OR module (mean)
        %   j = 6: pinwheel density per OD module (mean).
        pinw = [size(ORpinw,1); size(ORpinwp,1); size(ORpinwn,1)] * ...
            [1 1/M ORlM^2/M ODlM^2/M ORlm^2/M ODlm^2/M];
        pinwOD = zeros(3,1);			% To be filled in later
        
        stats.fig(150).pinw{ENcounter} = pinw;
        
        % Mark pinwheels in the contours (fig. 2) and OR angle map (fig. 3) too.
        if plotfig(2)
            set(0,'CurrentFigure',2);
            hold on;
            plot(ORpinwp(:,1),ORpinwp(:,2),'r+','MarkerSize',10);
            plot(ORpinwn(:,1),ORpinwn(:,2),'g+','MarkerSize',10);
            hold off;
        end
        if plotfig(3)
            set(0,'CurrentFigure',3);
            hold on;
            plot(ORpinwp(:,1),ORpinwp(:,2),'k+','MarkerSize',10);
            plot(ORpinwn(:,1),ORpinwn(:,2),'w+','MarkerSize',10);
            hold off;
        end
        
    end
    if any(plotfig([50:54 60 150:154 160])) || statsOnly
        % Locate the OD borders:
        ODborders = ENV1ODbord(G,mu(:,id.OD),v(id.OD,2:3));
    end
    
    % Plot pinwheels wrt OD borders for visual check:
    Figs = 50;
    if ishandle(7)
        Figsp = get(7,'Position'); Figsp =  Figsp(1:2);
    else
        Figsp = [scw sch]/4;
    end
    Figss = floor(min(scw,sch)/3);
    if plotfig(Figs)
        if firstone
            if ~ishandle(Figs)
                figure(Figs);
            end
            set(Figs,'Name','Pinwheels and OD borders');
        end
        set(0,'CurrentFigure',Figs); cla;
        tmp = get(Figs,'Position'); Figsp = tmp(1:2);
        myplot(G,bc,ones(M,1),'img',[1 0 1],[],Pi,[Figsp Figss]);
        hold on;
        plot(ODborders(:,1),ODborders(:,2),'k-','LineWidth',1);
        plot(ORpinwp(:,1),ORpinwp(:,2),'r*');
        plot(ORpinwn(:,1),ORpinwn(:,2),'bo');
        hold off;
        % There is no space in the figure title to put all statistics
        title(['#: '  num2str(pinw(1,1)) ...
            ' (+' num2str(pinw(2,1)) ...
            ', -' num2str(pinw(3,1)) ...
            ');    \rho_{OR} = ' num2str(pinw(1,3),3) ...
            '_{M},' num2str(pinw(1,5),3) ...
            '_{m};    \rho_{OD} = ' num2str(pinw(1,4),3),...
            '_{M},' num2str(pinw(1,6),3) '_{m}'],...
            'Visible','on','FontSize',9);
        drawnow;
        if ~isempty(ENdir)
            print(Figs,'-loose','-r150',['-d' EXT],sprintf('%s-fig%03d.%s',frame,Figs,EXT));
        end
    end
    
    % --- Figures 51-54: orientation pinwheels wrt ocular dominance borders
    
    % Compute histogram and statistics if required here or elsewhere
    if any(plotfig([50:54 60 150:154 160])) || statsOnly
        % Unweighted histogram of the distance between a pinwheel and its
        % closest OD border, for all pinwheels (using absolute frequencies)
        % for two cases:
        % 1. The actual pinwheels (separately for: all; positive; negative);
        % 2. The same number of pinwheels (both signs) but uniformly distributed
        %    at random on the same OD map, and repeated over several trials to
        %    get error bars (called "random (all), simulated" in the graph).
        %    I also plot (called "random (all), theory" in the graph) the
        %    theoretical distribution of distances for plane-wave OD borders of
        %    constant wavelength ODlm, which is uniform in [0,ODlm/4] for
        %    pinwheels uniformly randomly distributed. For general shapes of the
        %    OD borders, the distribution is complicated to obtain - thus my
        %    giving the simulated one too.
        %    Note that the fact that the ODborders are a sample of the
        %    underlying borders (typically spaced around 1 pixel) means that
        %    there is an artifactual scarcity of very short distances (only
        %    apparent for small ODlm).
        if any(plotfig([54 60 154])) && (npinw >= 1) || statsOnly
            Ntrials = NTrials;
        else
            Ntrials = 0;
        end
        
        % The index Ntrials+1 in dist(:,i) and disth(:,i) corresponds to the
        % actual pinwheels.
        if npinw >= 1
            dist = zeros(npinw,Ntrials+1);	% List of distances for each trial
            pinwR = zeros(npinw,2,Ntrials);
            disth = zeros(nbins,Ntrials+1);	% Histogram for each trial
            % Random pinwheels, independent of sign:
            for i=1:Ntrials
                ipinw = 1;
                while ipinw <= npinw
                    tmp = (rand(npinw,2)*diag(G-1))+1;
                    pick = find(Pi(sub2ind(G,round(tmp(:,1)),round(tmp(:,2)))) > 0);
                    npick = length(pick);
                    if npick > 0
                        npick = min(npick,npinw+1-ipinw);
                        rndpinw(ipinw:ipinw-1+npick,:) = tmp(pick(1:npick),:);
                        ipinw = ipinw + npick;
                    end
                end
                dist(:,i) = sqrt(min(...
                    ENsqdist(rndpinw+repmat(1/2,npinw,2),ODborders),...
                    [],2));
                pinwR(:,:,i) = rndpinw; 
            end
            % Actual pinwheels, independent of sign:
            dist(:,end) = sqrt(min(ENsqdist(ORpinw,ODborders),[],2));
            % Number of pinwheels right on OD borders
            pinwOD(1) = sum(dist(:,end) < OD_pinw_th);
            % Determine maximum distance to have all histograms over the same
            % domain:
            distM = max([dist(:,Ntrials+1);ODlm/4;ODlM/4;ORlm/4;ODlM/4]);		% No need to include distp, distn
            % Histogram and statistics for the actual pinwheels:
            [h,KLu,L2u,m,s,g1] = ...
                ENhist(dist(:,Ntrials+1),[],nbins,[0 distM],'absfreq');
            disth(:,Ntrials+1) = h(:,1);
        else
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hact = {h,KLu,L2u,m,s,g1}; hact = h(:,1);
        
        % Positive and negative pinwheels:
        if npinwp >= 1
            distp = sqrt(min(ENsqdist(ORpinwp,ODborders),[],2));
            pinwOD(2) = sum(distp < OD_pinw_th);
            [h,KLu,L2u,m,s,g1] = ENhist(distp,[],nbins,[0 distM],'absfreq');
        else
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hactp = {h,KLu,L2u,m,s,g1}; hactp = h(:,1);
        if npinwn >= 1
            distn = sqrt(min(ENsqdist(ORpinwn,ODborders),[],2));
            pinwOD(3) = sum(distn < OD_pinw_th);
            [h,KLu,L2u,m,s,g1] = ENhist(distn,[],nbins,[0 distM],'absfreq');
        else
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hactn = {h,KLu,L2u,m,s,g1}; hactn = h(:,1);
        
        % Set pinw array of fig. 160 with the number of pinwheels right on OD
        % borders (up to OD_pinw_th), same format as the pinw array of fig. 150:
        stats.fig(160).pinw{ENcounter} = pinwOD * ...
            [1 1/M ORlM^2/M ODlM^2/M ORlm^2/M ODlm^2/M];
        % Update title of fig. 50
        if plotfig(50)
            tmp = get(get(50,'CurrentAxes'),'Title');
            set(tmp,'String',...
                [get(tmp,'String') ';    #OD: '  num2str(pinwOD(1)) ...
                ' (+' num2str(pinwOD(2)) ', -' num2str(pinwOD(3)) ')']);
        end
        
        % Average histogram and statistics for the uniformly distributed
        % pinwheel sets:
        if Ntrials > 0
            % Histogram and statistics for the random pinwheels:
            for i=1:Ntrials
                [h,KLu,L2u,m,s,g1] = ENhist(dist(:,i),[],nbins,[0 distM],'absfreq');
                disth(:,i) = h(:,1);
            end
            have = mean(disth(:,1:Ntrials),2);		% Average histogram
            hstd = std(disth(:,1:Ntrials),1,2);		% Error bars
            [h,KLu,L2u,m,s,g1] = ENhist(Hact{1}(:,2),have,nbins,[0 distM],'absfreq');
            h(:,1) = h(:,1) / sum(h(:,1)) * npinw;
        else
            have = repmat(NaN,nbins,1); hstd = have;
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hrnd = {h,KLu,L2u,m,s,g1};
        
        % Plot the histograms of pinwheel distances and display statistics:
        Vars = {Hact,Hactp,Hactn,Hrnd};
        Varsn = length(Vars);
        Varsl = {'(actual, all)','(actual, +)','(actual, -)',...
            ['(random, all, ' num2str(Ntrials) ' trials)']};
        Figs = 51; Figs = Figs:Figs+Varsn-1;
        Figss = [560 420];
        Figsp = [repmat(scw+1-Figss(1)-2*wnb,Varsn,1) ...
            -linspace((Figss(2)+wnt+wnb)-sch-1,...
            2*(Figss(2)+wnt+wnb)-sch-1,Varsn)'];
        tmp = 1.05*max([hact;have+hstd]);
        for i=1:Varsn
            if any(plotfig([Figs(i) Figs(i)+100]))
                [h,KLu,L2u,m,s,g1] = deal(Vars{i}{:});
                stats.fig(Figs(i)+100).KLu{ENcounter} = KLu;
                stats.fig(Figs(i)+100).L2u{ENcounter} = L2u;
                stats.fig(Figs(i)+100).m{ENcounter} = m;
                stats.fig(Figs(i)+100).s{ENcounter} = s;
                stats.fig(Figs(i)+100).g1{ENcounter} = g1;
            end
            if plotfig(Figs(i))
                if firstone
                    if ~ishandle(Figs(i))
                        figure(Figs(i));
                    end
                    set(Figs(i),'Position',[Figsp(i,:) Figss],...
                        'Name',['Histogram of pinwheel-OD-border distances '...
                        Varsl{i}],...
                        'Color','w','PaperPositionMode','auto');
                end
                set(0,'CurrentFigure',Figs(i));
                if isnan(KLu)
                    clf; axis off;
                    text(0.5,0.5,'Not available: empty histogram',...
                        'HorizontalAlignment','center');
                else
                    if (i == Varsn) & plotfig(Figs(Varsn))
                        H = plot(h(:,2),[h(:,1) hact hactp hactn]');
                        hold on;
                        H1 = plot([0 distM],[1 1]*npinw*distM/(nbins*ODlm/4),':g');
                        errorbar(h(:,2),have,hstd,hstd,'b.');
                        % $$$ 	    errorbar(h(:,2),have,zeros(nbins,1),hstd,'b.');
                        hold off;
                        set(H,'LineWidth',2);
                        legend([H1;H],'random (all), theory','random (all), simulated',...
                            'actual (all)','actual (+)','actual (-)','Location','best');
                    else
                        bar(h(:,2),h(:,1),1);
                    end
                    hold on;
                    plot(ODlM/4*[1 1],[0 tmp],'--r');		% OD wavelength (mode)
                    plot(ORlM/4*[1 1],[0 tmp],'--m');		% OR wavelength (mode)
                    plot(ODlm/4*[1 1],[0 tmp],':r');		% OD wavelength (mean)
                    plot(ORlm/4*[1 1],[0 tmp],':m');		% OR wavelength (mean)
                    hold off;
                    text(ODlM/4,tmp/9,'OD\lambda_M/4','HorizontalAlignment','center');
                    text(ORlM/4,2*tmp/9,'OR\lambda_M/4','HorizontalAlignment','center');
                    text(ODlm/4,3*tmp/9,'OD\lambda_m/4','HorizontalAlignment','center');
                    text(ORlm/4,4*tmp/9,'OR\lambda_m/4','HorizontalAlignment','center');
                    set(gca,'XLim',[0 distM],'YLim',[0 tmp]);
                    xlabel('Distance in pixels');
                end
                title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
                    ';   \mu=' num2str(m,3) ' pixels;   \sigma=' num2str(s,3) ...
                    ' pixels;   \gamma_1=' num2str(g1,3)],'Visible','on');
                drawnow;
                if ~isempty(ENdir)
                    print(Figs(i),'-loose','-r150',['-d' EXT],...
                        sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
                end
            end
        end
    end

    if plotfig(60) || statsOnly % normalized fig. 54
        % The index Ntrials+1 in dist(:,i) and disth(:,i) corresponds to the
        % actual pinwheels.
        if npinw >= 1
            dist = zeros(npinw,Ntrials+1);	% List of distances for each trial
            disth = zeros(nbins,Ntrials+1);	% Histogram for each trial
            % Random pinwheels, independent of sign:
            for i=1:Ntrials
                dist(:,i) = normDist(pinwR(:,:,i)+repmat(1/2,npinw,2),ODborders, ODlm/4);
            end
            % Actual pinwheels, independent of sign:
            % dist(:,end) = normDist(ORpinw,ODborders,ODlm/4,50);
            dist(:,end) = normDist(ORpinw,ODborders,ODlm/4);
            % Histogram and statistics for the actual pinwheels:
            distM = 0.5;
            [h,KLu,L2u,m,s,g1] = ...
                ENhist(dist(:,Ntrials+1),[],nbins,[0 distM],'absfreq');
            disth(:,Ntrials+1) = h(:,1);
        else
            [h,KLu,L2u,m,s,g1] = deal(NaN(nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hact = {h,KLu,L2u,m,s,g1}; hact = h(:,1);
        
        % Average histogram and statistics for the uniformly distributed
        % pinwheel sets:
        if Ntrials > 0
            % Histogram and statistics for the random pinwheels:
            for i=1:Ntrials
                [h,KLu,L2u,m,s,g1] = ENhist(dist(:,i),[],nbins,[0 distM],'absfreq');
                disth(:,i) = h(:,1);
            end
            have = mean(disth(:,1:Ntrials),2);		% Average histogram
            hstd = std(disth(:,1:Ntrials),1,2);		% Error bars
            [h,KLu,L2u,m,s,g1] = ENhist(Hact{1}(:,2),have,nbins,[0 distM],'absfreq');
            h(:,1) = h(:,1) / sum(h(:,1)) * npinw;
        else
            have = NaN(nbins,1); hstd = have;
            [h,KLu,L2u,m,s,g1] = deal(NaN(nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hrnd = {h,KLu,L2u,m,s,g1};
        [amp, stats.normD_mode] = max(hact,[],1,'omitnan');
        stats.normD_overMean = amp/mean(h(:,1),'omitnan');
        stats.normD_mode = h(stats.normD_mode,2);
        
        if plotfig(60)
            % Plot the histograms of pinwheel distances and display statistics:
            Vars = {Hact,Hrnd};
            Varsn = length(Vars);
            Varsl = {'(actual, all)', ['(random, all, ' num2str(Ntrials) ' trials)']};
            Figss = [560 420];
            Figsp = [repmat(scw+1-Figss(1)-2*wnb,1,1) ...
                -linspace((Figss(2)+wnt+wnb)-sch-1,...
                2*(Figss(2)+wnt+wnb)-sch-1,1)'];
            tmp = 1.05*max([hact;have+hstd]);
            if ~ishandle(60)
                figure(60);
            end
            set(60,'Position',[Figsp Figss],...
                'Name','Histogram of pinwheel-OD-border norm. dist',...
                'Color','w','PaperPositionMode','auto');
            set(0,'CurrentFigure',60); cla;
            H = plot(h(:,2),[h(:,1) hact]');
            hold on;
            H1 = plot([0 distM],[1 1]*npinw/nbins,':g');
            errorbar(h(:,2),have,hstd,hstd,'b.');
            % $$$ 	    errorbar(h(:,2),have,zeros(nbins,1),hstd,'b.');
            hold on;
            set(H,'LineWidth',2);
            legend([H1;H],'random (all), theory','random (all), simulated', 'actual (all)','Location','best');
            set(gca,'XLim',[0 distM],'YLim',[0 tmp]);
            xlabel('norm. dist inbetween ODborder');
            drawnow;
            if ~isempty(ENdir)
                print(60,'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,60,EXT));
            end
        end
    end
    
    % --- Figures 55-59: orientation pinwheels pairwise distances
    
    % Compute histogram and statistics if required here or elsewhere
    if any(plotfig([55:59 155:159]))
        % Unweighted histogram of the distance between a pinwheel and its
        % closest pinwheel, for all pinwheels (using absolute frequencies)
        % for two cases:
        % 1. The actual pinwheels (separately for: any sign-any sign;
        %    positive-positive; negative-negative; positive-negative);
        % 2. The same number of pinwheels (both signs) but uniformly distributed
        %    at random on the same map, and repeated over several trials to
        %    get error bars (called "random (all/all), simulated" in the graph).
        %    I also plot (called "random (all/all), theory" in the graph) the
        %    theoretical distribution of distances between nearest-neighbouring
        %    pinwheels for an unbounded cortex with density d = npinw/M
        %    pinwheels per square pixel, which is the following Rayleigh pdf
        %    (Cressie93a, sec. 8.4):
        %      p(r) = 2*pi*d*r*exp(-pi*d*r^2)
        %    with
        %      mean m = 1/sqrt(4*d),
        %      standard deviation s = sqrt((4-pi)/(4*pi*d)) and
        %      Fisher skewness g1 = 2*(pi-3)*sqrt(pi)*(4-pi)^(-3/2).
        %    This assumes that the pinwheels are distributed over the real plane
        %    as a 2D Poisson process with intensity parameter d. For a bounded
        %    cortex the distribution is complicated to compute because of the
        %    border effects - thus my giving the simulated one. The theoretical
        %    pdf tends to have a higher peak and thinner right-tail, thus
        %    slightly underestimating the mean, standard deviation and skewness.
        if any(plotfig([59 159])) & (npinw > 1)
            Ntrials = NTrials;
        else
            Ntrials = 0;
        end
        
        % The index Ntrials+1 in dist(:,i) and disth(:,i) corresponds to the
        % actual pinwheels.
        if npinw > 1
            dist = zeros(npinw,Ntrials+1);	% List of distances for each trial
            disth = zeros(nbins,Ntrials+1);	% Histogram for each trial
            % Random pinwheels, independent of sign:
            for i=1:Ntrials
                tmp = ENsqdist(rand(npinw,2)*diag(G)+repmat(1/2,npinw,2));
                ltmp = size(tmp,1); tmp(sub2ind(size(tmp),1:ltmp,1:ltmp)) = NaN;
                dist(:,i) = sqrt(min(tmp,[],2));
            end
            % Actual pinwheels, independent of sign:
            tmp = ENsqdist(ORpinw);
            ltmp = size(tmp,1); tmp(sub2ind(size(tmp),1:ltmp,1:ltmp)) = NaN;
            dist(:,end) = sqrt(min(tmp,[],2));
            % Determine maximum distance to have all histograms over the same
            % domain:
            distM = max(dist(:));		% No need to include distp, distn
            % Histogram and statistics for the actual pinwheels:
            [h,KLu,L2u,m,s,g1] = ...
                ENhist(dist(:,Ntrials+1),[],nbins,[0 distM],'absfreq');
            disth(:,Ntrials+1) = h(:,1);
        else
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hact = {h,KLu,L2u,m,s,g1}; hact = h(:,1);
        
        % Positive and negative pinwheels:
        if npinwp > 1
            tmp = ENsqdist(ORpinwp);
            ltmp = size(tmp,1); tmp(sub2ind(size(tmp),1:ltmp,1:ltmp)) = NaN;
            distp = sqrt(min(tmp,[],2));
            [h,KLu,L2u,m,s,g1] = ENhist(distp,[],nbins,[0 distM],'absfreq');
        else
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hactp = {h,KLu,L2u,m,s,g1}; hactp = h(:,1);
        if npinwn > 1
            tmp = ENsqdist(ORpinwn);
            ltmp = size(tmp,1); tmp(sub2ind(size(tmp),1:ltmp,1:ltmp)) = NaN;
            distn = sqrt(min(tmp,[],2));
            [h,KLu,L2u,m,s,g1] = ENhist(distn,[],nbins,[0 distM],'absfreq');
        else
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hactn = {h,KLu,L2u,m,s,g1}; hactn = h(:,1);
        if min(npinwp,npinwn) < 1
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        else
            tmp = ENsqdist(ORpinwp,ORpinwn);
            distpn = sqrt([min(tmp,[],2);min(tmp',[],2)]);
            [h,KLu,L2u,m,s,g1] = ENhist(distpn,[],nbins,[0 distM],'absfreq');
        end
        Hactpn = {h,KLu,L2u,m,s,g1}; hactpn = h(:,1);
        
        % Average histogram and statistics for the uniformly distributed
        % pinwheel sets:
        if Ntrials > 0
            % Histogram and statistics for the random pinwheels:
            for i=1:Ntrials
                [h,KLu,L2u,m,s,g1] = ENhist(dist(:,i),[],nbins,[0 distM],'absfreq');
                disth(:,i) = h(:,1);
            end
            have = mean(disth(:,1:Ntrials),2);		% Average histogram
            hstd = std(disth(:,1:Ntrials),1,2);		% Error bars
            [h,KLu,L2u,m,s,g1] = ENhist(h(:,2),have,nbins,[0 distM],'absfreq');
            h(:,1) = h(:,1) / sum(h(:,1)) * npinw;
        else
            have = repmat(NaN,nbins,1); hstd = have;
            [h,KLu,L2u,m,s,g1] = deal(repmat(NaN,nbins,4),NaN,NaN,NaN,NaN,NaN);
        end
        Hrnd = {h,KLu,L2u,m,s,g1};
        
        % Plot the histograms of pinwheel distances and display statistics:
        Vars = {Hact,Hactp,Hactn,Hactpn,Hrnd};
        Varsn = length(Vars);
        Varsl = {'(actual, all/all)','(actual, +/+)','(actual, -/-)',...
            '(actual, +/-)',...
            ['(random, all/all, ' num2str(Ntrials) ' trials)']};
        Figs = 55; Figs = Figs:Figs+Varsn-1;
        Figss = [560 420];
        Figsp = [repmat(scw+1-2*Figss(1)-4*wnb,Varsn,1) ...
            -linspace((Figss(2)+wnt+wnb)-sch-1,...
            2*(Figss(2)+wnt+wnb)-sch-1,Varsn)'];
        tmp = 1.05*max([hact;have+hstd]);
        for i=1:Varsn
            if any(plotfig([Figs(i) Figs(i)+100]))
                [h,KLu,L2u,m,s,g1] = deal(Vars{i}{:});
                stats.fig(Figs(i)+100).KLu{ENcounter} = KLu;
                stats.fig(Figs(i)+100).L2u{ENcounter} = L2u;
                stats.fig(Figs(i)+100).m{ENcounter} = m;
                stats.fig(Figs(i)+100).s{ENcounter} = s;
                stats.fig(Figs(i)+100).g1{ENcounter} = g1;
            end
            if plotfig(Figs(i))
                if firstone
                    if ~ishandle(Figs(i))
                        figure(Figs(i));
                    end
                    set(Figs(i),'Position',[Figsp(i,:) Figss],...
                        'Name',...
                        ['Histogram of nearest-pinwheel-pair distances '...
                        Varsl{i}],...
                        'Color','w','PaperPositionMode','auto');
                end
                set(0,'CurrentFigure',Figs(i));
                if isnan(KLu)
                    clf; axis off;
                    text(0.5,0.5,'Not available: empty histogram',...
                        'HorizontalAlignment','center');
                else
                    if (i == Varsn) & plotfig(Figs(Varsn))
                        H = plot(h(:,2),[h(:,1) hact hactp hactn hactpn]');
                        hold on;
                        raylx = linspace(0,distM,100);
                        rayly = 2*pi*npinw/M*raylx.*exp(-pi*npinw/M*raylx.^2);
                        H1 = plot(raylx,rayly*npinw*distM/nbins,'m:');
                        errorbar(h(:,2),have,hstd,hstd,'b.');
                        % $$$ 	    errorbar(h(:,2),have,zeros(nbins,1),hstd,'b.');
                        hold off;
                        set(H,'LineWidth',2);
                        legend([H1;H],'random (all/all), theory',...
                            'random (all/all), simulated','actual (all/all)',...
                            'actual (+/+)','actual (-/-)','actual (+/-)','Location','best');
                    else
                        bar(h(:,2),h(:,1),1);
                    end
                    hold on;
                    plot(ODlM/4*[1 1],[0 tmp],'r--');		% OD wavelength (mode)
                    plot(ORlM/4*[1 1],[0 tmp],'r--');		% OR wavelength (mode)
                    plot(ODlm/4*[1 1],[0 tmp],'m--');		% OD wavelength (mean)
                    plot(ORlm/4*[1 1],[0 tmp],'m--');		% OR wavelength (mean)
                    hold off;
                    text(ODlM/4,-tmp/9,'OD\lambda_M/4','HorizontalAlignment','center');
                    text(ORlM/4,-tmp/9,'OR\lambda_M/4','HorizontalAlignment','center');
                    text(ODlm/4,-tmp/9,'OD\lambda_m/4','HorizontalAlignment','center');
                    text(ORlm/4,-tmp/9,'OR\lambda_m/4','HorizontalAlignment','center');
                    set(gca,'XLim',[0 distM],'YLim',[0 tmp]);
                    xlabel('Distance in pixels');
                end
                title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
                    ';   \mu=' num2str(m,3) ' pixels;   \sigma=' num2str(s,3) ...
                    ' pixels;   \gamma_1=' num2str(g1,3)],'Visible','on');
                drawnow;
                if ~isempty(ENdir)
                    print(Figs(i),'-loose','-r150',['-d' EXT],...
                        sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
                end
            end
        end
    end
    
    % --- Figures 70-75: gradient correlations
    % The interesting variables to consider pairwise correlations of are
    % gVF, gOD, gORt & ORr. I don't consider gORr because it behaves
    % similarly to ORr: roughly constant except at singularities.
    % gVFx and gVFy behave similarly to each other, and similarly to gVF,
    % although the latter has a less "discretised" aspect.
    Vars = [gr(:,[8 id.OD id.OR]) mu(:,id.ORr)];	% gVF,gOD,gORt,ORr
    Vars(:,3) = Vars(:,3)*180/pi;		% Pass gORt to degrees/pixel
    VarsL = {'Visual field gradient',...
        'Ocular dominance gradient',...
        'Orientation angle gradient',...
        'Orientation selectivity'};
    Varsl = {'gVF','gOD','gORt','ORr'};
    Varsax = [min(Vars,[],1);max(Vars,[],1)];
    Varsax = Varsax + [-1;1]*diff(Varsax,1,1)/10;
    % $$$   Vars = gr(:,[8 3 6 7]);		% gVF,gOD,gORt,gORr
    % $$$       'Orientation selectivity gradient'};
    % $$$   Varsl = {'gVF','gOD','gORt','gORr'};
    Varsn = size(Vars,2); Varsn = Varsn*(Varsn-1)/2;
    % Compute correlations if required here or elsewhere
    if any(plotfig([70:70+Varsn-1 170:171]))
        [C,c,linregr] = v1Corr(Vars);
        % Both C and c are symmetric 4x4 matrices with components as follows:
        %              gVF         gOD         gORt         ORr
        %
        % gVF          ----        ----        ----        ----
        % gOD         C(2,1)       ----        ----        ----
        % gORt        C(3,1)      C(3,2)       ----        ----
        % ORr         C(4,1)      C(4,2)      C(4,3)       ----
        % where C is Pearson's correlation and c the angle cosine between the
        % corresponding variables. E.g. c(4,2) is the angle cosine between ORr
        % and gOD.
        %
        % Note that C(3,1) = correlation between gVF and gORt is related, but
        % not equivalent, to what DasGilbert97a reported in their fig. 2a-b.
        % The differences are:
        % - Since the elastic net has a constant (and irrelevant) RF size,
        %   normalising by it does not make any difference. Likewise, the
        %   cortical distance is constant (since we use true gradients).
        % - DasGilbert97a used directional gradients, i.e., d(VF1,VF2)/d(1,2),
        %   where 1 and 2 represent two cortical locations --often widely
        %   separated--- and d() the Euclidean distance. We use true gradients
        %   (up to the pixel resolution).
        % - DasGilbert97a seem to have mapped all OR differences to [0,90].
        if plotfig(170)
            stats.fig(170).C{ENcounter} = C;
        end
        if plotfig(171)
            stats.fig(171).c{ENcounter} = c;
        end
        
        % Mark pinwheels (ORpinwi*: pinwheel locations as linear indices):
        if npinwp < 1
            ORpinwip = [];
        else
            ORpinwip = round(ORpinwp);
            ORpinwip = [ORpinwip sub2ind(G,ORpinwip(:,1),ORpinwip(:,2))];
        end
        if npinwn < 1
            ORpinwin = [];
        else
            ORpinwin = round(ORpinwn);
            ORpinwin = [ORpinwin sub2ind(G,ORpinwin(:,1),ORpinwin(:,2))];
        end
        
        % Plot figures
        Figs = 70; Figsp = [1 1]; Figss = [350 300];
        for i=2:4
            for j=1:i-1
                if plotfig(Figs)
                    if firstone
                        if ~ishandle(Figs)
                            figure(Figs);
                        end
                        set(Figs,'Position',...
                            [Figsp+[j-1 4-i].*(Figss+[2*wnb wnt+wnb]) Figss],...
                            'Name',...
                            ['Gradient scatterplot: ' Varsl{j} ' vs ' Varsl{i}],...
                            'Color','w','PaperPositionMode','auto');
                    end
                    set(0,'CurrentFigure',Figs);
                    plot(Vars(:,i),Vars(:,j),'m.','MarkerSize',3,'MarkerFaceColor','r');
                    % Mark pinwheels
                    hold on;
                    if npinwp >= 1
                        plot(Vars(ORpinwip(:,3),i),Vars(ORpinwip(:,3),j),'ro');
                    end
                    if npinwn >= 1
                        plot(Vars(ORpinwin(:,3),i),Vars(ORpinwin(:,3),j),'bo');
                    end
                    % Draw regression lines
                    tmpm = linregr(i,j).mean;
                    tmpl = norm(diff([Varsax(:,i) Varsax(:,j)],1));
                    tmpv = linregr(i,j).vxy;
                    tmp = [tmpm tmpm] + [-tmpv tmpv]*tmpl/norm(tmpv);
                    h1 = plot(tmp(1,:),tmp(2,:),'g-');	% Regression i -> j
                    tmpv = linregr(i,j).vyx;
                    tmp = [tmpm tmpm] + [-tmpv tmpv]*tmpl/norm(tmpv);
                    h2 = plot(tmp(1,:),tmp(2,:),'c-');	% Regression j -> i
                    tmpv = linregr(i,j).v;
                    tmp = [tmpm tmpm] + [-tmpv tmpv]*tmpl/norm(tmpv);
                    h3 = plot(tmp(1,:),tmp(2,:),'k-');	% Regression i, j
                    hold off;
                    axis([Varsax(:,i);Varsax(:,j)]');
                    xlabel(VarsL{i}); ylabel(VarsL{j});
                    legend([h1 h2 h3],'X \rightarrow Y','Y \rightarrow X','X,Y','Location','best');
                    title(['Pearson''s correlation r = ' num2str(C(i,j),3) ...
                        ';   Angle cosine c = ' num2str(c(i,j),3)],'Visible','on');
                    drawnow;
                    if ~isempty(ENdir)
                        print(Figs,'-loose','-r150',['-d' EXT],...
                            sprintf('%s-fig%03d.%s',frame,Figs,EXT));
                    end
                end
                Figs = Figs + 1;
            end
        end
    end
    
    % --- Figure 76: unweighted histogram of the product of moduli of OD
    %     and OR gradients.
    % This is another way of analysing the degree of match of OD borders
    % and OR pinwheels; but it is only useful to compare with other
    % histograms making sure to note how their X axes differ.
    % This histogram:
    % - Is a derivative of the correlation plot of gOD vs gORt (fig. 32).
    % - Is related to, but different, from the cosine of the angle between
    %   gOD and gORt.
    % It isn't very useful; fig. 72 is better.
    % Plot the histogram of gOD.gORt and display statistics of it:
    Figs = 76; Figsp = [scw sch]/4; Figss = [560 420];
    % Compute histogram and statistics if required here or elsewhere
    if any(plotfig([Figs Figs+100]))
        [h,KLu,L2u,m,s,g1] = ENhist(gr(:,id.OD).*gr(:,id.OR),[],nbins);
        stats.fig(Figs+100).KLu{ENcounter} = KLu;
        stats.fig(Figs+100).L2u{ENcounter} = L2u;
        stats.fig(Figs+100).m{ENcounter} = m;
        stats.fig(Figs+100).s{ENcounter} = s;
        stats.fig(Figs+100).g1{ENcounter} = g1;
    end
    if plotfig(Figs)
        if firstone
            if ~ishandle(Figs)
                figure(Figs);
            end
            set(Figs,'Position',[Figsp Figss],...
                'Name','Histogram of gOD.gORt',...
                'Color','w','PaperPositionMode','auto');
        end
        set(0,'CurrentFigure',Figs);
        bar(h(:,2),h(:,1),1);
        xlabel('Product of the gradient moduli of OD and OR');
        title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
            ';   \mu=' num2str(m,3) ';   \sigma=' num2str(s,3) ...
            ';   \gamma_1=' num2str(g1,3)],'Visible','on');
        drawnow;
        if ~isempty(ENdir)
            print(Figs,'-loose','-r150',['-d' EXT],sprintf('%s-fig%03d.%s',frame,Figs,EXT));
        end
    end
    
    
    % --- Figures 80-89: gradient direction of VFx,VFy,OD,ORt,ORr in [-90,90]
    %     degrees.
    % The gradient direction is the angle t that the stripes would make with
    % the vertical, defined so that (when plotted with myplot):
    %   t =       -pi/2   -pi/4     0     pi/4    pi/2
    %   stripes =   -       /       |       \       -
    Varsi = [1 2 3 6 7];
    Varsn = length(Varsi);
    Vars = mu(:,Varsi);				% VFx,VFy,OD,ORt,ORr
    Varsl = {'VFx','VFy','OD','ORt','ORr'};
    Figs = 80; Figs = Figs:Figs+2*Varsn-1;
    Figsp = [(linspace(1,scw/2,Varsn))' repmat(sch/2,Varsn,1);
        (linspace(1,scw/2,Varsn))' repmat(sch/4,Varsn,1)];
    Figss = [560 420]*2;
    angh = zeros(nbins,2*Varsn);			% For figs. 90-91
    
    for i=1:Varsn
        % Interior points
        % Compute histogram and statistics if required here or elsewhere
        if any(plotfig([Figs(i) Figs(i)+100 90]))
            angI = gt(I_ind,Varsi(i))*180/pi;
            tmp = find(angI < 0); angI(tmp) = angI(tmp) + 180; angI = angI - 90;
            % Histogram weighting angles according to their gradient modulus:
            [h,KLu,L2u,m,s,g1] = ENhist(angI,gr(I_ind,Varsi(i)),nbins,[-90 90]);
            angh(:,i) = h(:,1);
            stats.fig(Figs(i)+100).KLu{ENcounter} = KLu;
            stats.fig(Figs(i)+100).L2u{ENcounter} = L2u;
            stats.fig(Figs(i)+100).m{ENcounter} = m;
            stats.fig(Figs(i)+100).s{ENcounter} = s;
            stats.fig(Figs(i)+100).g1{ENcounter} = g1;
        end
        if plotfig(90)
            hX = h(:,2);
        end
        % Plot this figure
        if plotfig(Figs(i))
            if firstone
                if ~ishandle(Figs(i))
                    figure(Figs(i));
                end
                set(Figs(i),'Position',[Figsp(i,:) Figss],...
                    'Name',['Histogram of gradient direction of ' Varsl{i} ...
                    ' (interior points)'],...
                    'Color','w','PaperPositionMode','auto');
            end
            set(0,'CurrentFigure',Figs(i));
            plot([-90 90],[1 1]/90/2,'r--');		% Uniform distribution
            hold on; bar(h(:,2),h(:,1),1); hold off;
            set(gca,'XLim',[-90 90],'XTick',-90:30:90);
            xlabel('Gradient direction in degrees (\circ)');
            title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
                ';   \mu=' num2str(m,3) '\circ;   \sigma=' num2str(s,3) ...
                '\circ;   \gamma_1=' num2str(g1,3)],'Visible','on');
            drawnow;
            if ~isempty(ENdir)
                print(Figs(i),'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
            end
        end
        
        % Boundary points
        % Compute histogram and statistics if required here or elsewhere
        if any(plotfig([Figs(i)+Varsn Figs(i)+Varsn+100 91]))
            if B_width > 0
                angB = gt(B_ind,Varsi(i))*180/pi;
                tmp = find(angB < 0); angB(tmp) = angB(tmp) + 180; angB = angB - 90;
                [h,KLu,L2u,m,s,g1] = ENhist(angB,gr(B_ind,Varsi(i)),nbins,[-90 90]);
            else
                [h,KLu,L2u,m,s,g1] = deal(NaN);
            end
            angh(:,i+Varsn) = h(:,1);
            stats.fig(Figs(i)+Varsn+100).KLu{ENcounter} = KLu;
            stats.fig(Figs(i)+Varsn+100).L2u{ENcounter} = L2u;
            stats.fig(Figs(i)+Varsn+100).m{ENcounter} = m;
            stats.fig(Figs(i)+Varsn+100).s{ENcounter} = s;
            stats.fig(Figs(i)+Varsn+100).g1{ENcounter} = g1;
        end
        % Plot this figure
        if plotfig(Figs(i)+Varsn)
            if firstone
                if ~ishandle(Figs(i)+Varsn)
                    figure(Figs(i)+Varsn);
                end
                set(Figs(i)+Varsn,'Position',[Figsp(i+Varsn,:) Figss],...
                    'Name',...
                    ['Histogram of gradient direction of ' Varsl{i} ...
                    ' (boundary points)'],...
                    'Color','w',...
                    'PaperPositionMode','auto');
            end
            set(0,'CurrentFigure',Figs(i)+Varsn);
            if B_width > 0
                plot([-90 90],[1 1]/90/2,'r--');        % Uniform distribution
                hold on; bar(h(:,2),h(:,1),1); hold off;
                set(gca,'XLim',[-90 90],'XTick',-90:30:90);
                xlabel('Gradient direction in degrees (\circ)');
            else
                axis off; cla;
                text(0.5,0.5,'Not available: the boundary width is zero',...
                    'HorizontalAlignment','center');
            end
            title(['KL(u)=' num2str(KLu,3) ';   L_2(u)=' num2str(L2u,3) ...
                ';   \mu=' num2str(m,3) '\circ;   \sigma=' num2str(s,3) ...
                '\circ;   \gamma_1=' num2str(g1,3)],'Visible','on');
            drawnow;
            if ~isempty(ENdir)
                print(Figs(i)+Varsn,'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i)+Varsn,EXT));
            end
        end
    end
    
    
    % --- Figures 90-91: combination of figs. 80-89.
    % Uses Figsp, Varsn, Varsl and hX from the previous figure.
    Figs = 90:91;
    VarsL = {'(interior points)','(boundary points)'};
    Figss = [560 420]*2;
    Figsp = Figsp([1 1+Varsn],:);
    for i=1:2
        if plotfig(Figs(i))
            if firstone
                if ~ishandle(Figs(i))
                    figure(Figs(i));
                end
                set(Figs(i),'Position',[Figsp(i,:) Figss],...
                    'Name',...
                    ['Combined histograms of gradient direction ' VarsL{i}],...
                    'Color','w','PaperPositionMode','auto');
            end
            set(0,'CurrentFigure',Figs(i));
            if (B_width > 0) | (i == 1)
                plot([-90 90],[1 1]/90/2,'r--');		% Uniform distribution
                hold on; H = plot(hX,angh(:,(1:Varsn)+(i-1)*Varsn)); hold off;
                set(gca,'XLim',[-90 90],'XTick',-90:30:90,'YLim',[0 1.1*max(angh(:))]);
                xlabel('Gradient direction in degrees (\circ)');
                set(H,'LineWidth',2);
                legend(H,Varsl{:},'Location','best');
            else
                axis off; cla;
                text(0.5,0.5,'Not available: the boundary width is zero',...
                    'HorizontalAlignment','center');
            end
            drawnow;
            if ~isempty(ENdir)
                print(Figs(i),'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
            end
        end
    end
    
    
    % --- Figures 95-97: gradient plots
    % Overlay the contours of OD and OR (as in fig. 2) on a grayscale plot
    % (actually, using a see-through colormap) of the different gradient:
    % gVF (fig. 95), gOD (fig. 96) and gOR (fig. 97).
    %
    % Fig. 95 is similar to fig. 3d in DasGilbert97a.
    Varsi = [8 3 6];
    Varsn = length(Varsi);
    Vars = gr(:,Varsi);			% gVF,gOD,gORt
    Vars(:,3) = Vars(:,3)*180/pi;		% Pass gORt to degrees/pixel
    Varsl = {'gVF','gOD','gORt'};
    Figs = 95; Figs = Figs:Figs+Varsn-1;
    Figsp = [repmat(3*scw/4,Varsn,1) ...
        (linspace(50,50+(Varsn-1)*(Figl+2*wnb+wnt+fgt+2*fgb),Varsn))'];
    Figss = Figl;
    cmap = jet(256); cmap = cmap(32:end-31,:);
    for i=1:Varsn
        if plotfig(Figs(i))
            if firstone
                if ~ishandle(Figs(i))
                    figure(Figs(i));
                end
                set(Figs(i),'Position',[Figsp(i,:) 10 10],...
                    'Name',['OD/OR contours over ' Varsl{i}]);
            end
            set(0,'CurrentFigure',Figs(i)); cla;
            tmp = get(Figs(i),'Position'); Figsp(i,:) = tmp(1:2);
            myplot(G,bc,Vars,GRplotv,[i 0 max(Vars(:,i))],T,Pi,[Figsp(i,:) Figss]);
            myplot(G,bc,mu,ODplotv,v(id.OD,:),T,Pi,[Figsp(i,:) Figss]);
            myplot(G,bc,mu,ORplotv,v(id.OR,:),T,Pi,[Figsp(i,:) Figss]);
            title(Kstr,'Visible','on'); drawnow;
            if ~isempty(ENdir)
                print(Figs(i),'-loose','-r150',['-d' EXT],...
                    sprintf('%s-fig%03d.%s',frame,Figs(i),EXT));
            end
        end
    end
    if firstone & any(plotfig(Figs))
        ENcolorbar([],Figs(end)+1,min(find(plotfig(Figs)))+Figs(1)-1); drawnow;
        if ~isempty(ENdir)
            print(Figs(end)+1,'-loose','-r150',['-d' EXT],...
			sprintf('%s-fig%03d.%s',frame,Figs(end)+1,EXT));
        end
    end
    
    
    % --- Other plots (myplot)
    % Other gradients
    
    % ------------------------------------------------------------------------
    
    if strcmp(opt,'pause')
        pause
    end
    
    firstone = logical(0);
end


% --------------------------------------------------------------------------
% Time-course figures: summary statistics as a function of the iteration
% number.

% Time-courses are only plotted for more than one frame
if length(whichones) == 1
    return;
end

Ks = []; its = zeros(length(ENlist),1);
for ENcounter = 2:length(ENlist)
    its(ENcounter) = its(ENcounter-1) + size(ENlist(ENcounter).stats.K,1);
    Ks = [Ks;ENlist(ENcounter).stats.K];
end
its(1) = NaN;
lKs = length(Ks);
stats.Ks = Ks;
stats.its = its;

% Horizontal axis properties (ticks for the iteration number), the same for
% all figures below
Xval = its(whichones); XM = max(Xval); Xm = min(Xval);
Xr = XM - Xm;
Xax = round([Xm XM] + 0.07*Xr*[-1 1]);
if (min(whichones)==1) | (Xax(1)<0)
    Xax(1) = 1;
end
XT = Xax(1):ceil(diff(Xax)/10):Xax(2);
XT = XT(XT<=lKs);


% --- Figures 100-101: energy and computation time (from ENV1replay2).
Figs = 100:101;
for i=find(plotfig(Figs))
    ENV1replay2(G,bc,ENlist,v,1,T(:,1:5),Pi,murange,tens,[],Figs(i));
    if ~isempty(ENdir)
        print(Figs(i),'-loose','-r150',['-d' EXT],...
            sprintf([ENdir,'/fig%03d.%s'],Figs(i),EXT));
    end
end


% --- Figures 120-124,130-133,151-159,176,180-189: histogram statistics for
%     figs. 20-24,30-33,51-59,76,80-89 (KLu,L2u,m,s,g1).
Figs = [120:124 130:133 151:159 176 180:189];
Figsl = {'visual field X',...
    'visual field Y',...
    'ocular dominance',...
    'orientation angle',...
    'orientation selectivity',...
    'OD/OR crossing angles (interior points)',...
    'OD/OR crossing angles (boundary points)',...
    'OD/boundary crossing angles',...
    'ORt/boundary crossing angles',...
    'pinw/OD-border distances (actual, all)',...
    'pinw/OD-border distances (actual, +)',...
    'pinw/OD-border distances (actual, -)',...
    ['pinw/OD-border distances (random, all, '...
    num2str(NTrials) ' trials)'],...
    'pinw/pinw closest distances (actual, all/all)',...
    'pinw/pinw closest distances (actual, +/+)',...
    'pinw/pinw closest distances (actual, -/-)',...
    'pinw/pinw closest distances (actual, +/-)',...
    ['pinw/pinw closest distances (random, all/all, '...
    num2str(NTrials) ' trials)'],...
    'product of the gradient moduli of OD and OR',...
    'gradient direction (VFx) (interior points)',...
    'gradient direction (VFy) (interior points)',...
    'gradient direction (OD) (interior points)',...
    'gradient direction (ORt) (interior points)',...
    'gradient direction (ORr) (interior points)',...
    'gradient direction (VFx) (boundary points)',...
    'gradient direction (VFy) (boundary points)',...
    'gradient direction (OD) (boundary points)',...
    'gradient direction (ORt) (boundary points)',...
    'gradient direction (ORr) (boundary points)'};
Figsunitsl = {'','','','','','','','','','','','','',...
    '','','','','','','','','','','','','','','',''};
Figsunitsr = {'','','','','',' (\circ)',' (\circ)',' (\circ)',' (\circ)',...
    ' (pixels)',' (pixels)',' (pixels)',' (pixels)',' (pixels)',...
    ' (pixels)',' (pixels)',' (pixels)',' (pixels)','',...
    ' (\circ)',' (\circ)',' (\circ)',' (\circ)',' (\circ)',...
    ' (\circ)',' (\circ)',' (\circ)',' (\circ)',' (\circ)'};
Figss = [560 420]*2;
Figsp = ...
    [(linspace(1,scw/2,5))' repmat(sch/2,5,1);...
    linspace(1,1+Figss(1),2)' repmat(sch+1-Figss(2)-wnt-wnb,2,1);...
    linspace(1,1+Figss(1)+2*wnb,2)' repmat(sch+1-2*(Figss(2)+wnt+wnb),2,1);...
    repmat(scw+1-Figss(1)-2*wnb,4,1) ...
    -linspace((Figss(2)+wnt+wnb)-sch-1,2*(Figss(2)+wnt+wnb)-sch-1,4)'
    repmat(scw+1-2*Figss(1)-4*wnb,5,1) ...
    -linspace((Figss(2)+wnt+wnb)-sch-1,2*(Figss(2)+wnt+wnb)-sch-1,5)';...
    [scw sch]/4;...
    (linspace(1,scw/2,5))' repmat(sch/4,5,1);...
    (linspace(1,scw/2,5))' repmat(sch/4,5,1)];
for i=find(plotfig(Figs))
    % Turn each cell array in stats.fig into an array:
    KLu = cell2mat(stats.fig(Figs(i)).KLu(whichones));
    L2u = cell2mat(stats.fig(Figs(i)).L2u(whichones));
    m = cell2mat(stats.fig(Figs(i)).m(whichones));
    s = cell2mat(stats.fig(Figs(i)).s(whichones));
    g1 = cell2mat(stats.fig(Figs(i)).g1(whichones));
    % Compute axes (left-side and right-side)
    lvars = [KLu L2u g1]; lcol = 'k';
    laxM = max(lvars); laxm = min(lvars); laxr = laxM - laxm;
    lax = [laxm laxM] + 0.05*laxr*[-1 1]; laxt = linspace(lax(1),lax(2),5);
    laxt = unique(str2num(num2str(laxt,2))); lax = laxt([1 end]);	% Round numbers
    rvars = [m s]; rcol = 'b';
    raxM = max(rvars); raxm = min(rvars); raxr = raxM - raxm;
    rax = [raxm raxM] + 0.05*raxr*[-1 1]; raxt = linspace(rax(1),rax(2),5);
    raxt = unique(str2num(num2str(raxt,2))); rax = raxt([1 end]);	% Round numbers
    % Plot this figure
    set(figure(Figs(i)),'Position',[Figsp(i,:) Figss],...
        'Name',['Histogram stats of ' Figsl{i}],...
        'Color','w','PaperPositionMode','auto');
    clf; hold on;
    % Plot each array
    if ismember(Figs(i),[131 185:189]) & (B_width <= 0)
        axis off; cla;
        text(0.5,0.5,'Not available: the boundary width is zero',...
            'HorizontalAlignment','center');
    else
        [AXa,H1,H2] = plotyy(Xval,KLu,Xval,m,'plot');
        set(AXa(1),'XLim',Xax,'XTick',[],'YLim',lax,'YTick',[]);
        set(AXa(2),'XLim',Xax,'XTick',[],'YLim',rax,'YTick',[]);
        set(H1,'LineStyle','-','Color',lcol)
        set(H2,'LineStyle','-','Color',rcol)
        [AXb,H3,H4] = plotyy(Xval,L2u,Xval,s,'plot');
        set(AXb(1),'XLim',Xax,'XTick',[],'YLim',lax,'YTick',[]);
        set(AXb(2),'XLim',Xax,'XTick',[],'YLim',rax,'YTick',[]);
        set(H3,'LineStyle','--','Color',lcol)
        set(H4,'LineStyle','--','Color',rcol)
        [AXc,H5,H6] = plotyy(Xval,g1,Xval,NaN,'plot');
        set(AXc(1),'XLim',Xax,'XTick',XT,...
            'YLim',lax,'YTick',laxt,'Box','on');
        set(AXc(2),'XLim',Xax,'XTick',[],...
            'YLim',rax,'YTick',raxt);
        set(H5,'LineStyle',':','Color',lcol)
        hold off;
        set(Figs(i),'CurrentAxes',AXc(2)); Figsax = [Xax rax];
        set(get(AXc(2),'YLabel'),...
            'String',['\mu, \sigma' Figsunitsr{i}],'Color',rcol);
        set(AXc(2),'YColor',rcol);
        set(get(AXc(1),'XLabel'),'String','Iteration');
        set(get(AXc(1),'YLabel'),...
            'String',['KL(u), L_2(u), \gamma_1' Figsunitsl{i}],'Color',lcol);
        set(AXc(1),'YColor',lcol);
        text(XT,repmat(Figsax(4)+0.025*diff(Figsax(3:4)),size(XT)),...
            num2str(Ks(XT),3),'HorizontalAlignment','center','FontSize',6);
        text(mean(Figsax(1:2)),Figsax(4)+0.065*diff(Figsax(3:4)),'K');
        legend([H1 H3 H5 H2 H4],'KL(u)','L2(u)','\gamma_1','\mu','\sigma','Location','best');
    end
    drawnow;
    if ~isempty(ENdir)
        print(Figs(i),'-loose','-r150',['-d' EXT],...
            sprintf([ENdir,'/fig%03d.%s'],Figs(i),EXT));
    end
end


% --- Figures 140-142: Fourier spectrum statistics for figs. 40-41
%     (lambdaM,alphaM,lambdam) and combination.
Figs = [140:141];
Figsl = {'OD','ORt'};
Figsunitsl = {' (pixels)',' (pixels)'};
Figsunitsr = {' (\circ)',' (\circ)'};
Figss = [560 420]*2;
Figsp = [1 1;1+Figss(1) 1];
for i=find(plotfig(Figs))
    % Turn each cell array in stats.fig into an array:
    lambdaM = cell2mat(stats.fig(Figs(i)).lambdaM(whichones));
    alphaM = cell2mat(stats.fig(Figs(i)).alphaM(whichones));
    lambdam = cell2mat(stats.fig(Figs(i)).lambdam(whichones));
    % Compute axes (left-side and right-side)
    lvars = [lambdaM lambdam]; lcol = 'k';
    laxM = max(lvars); laxm = min(lvars); laxr = laxM - laxm;
    lax = [laxm laxM] + 0.05*laxr*[-1 1]; laxt = linspace(lax(1),lax(2),5);
    laxt = unique(str2num(num2str(laxt,2))); lax = laxt([1 end]);	% Round numbers
    rvars = [alphaM]; rcol = 'b';
    raxM = max(rvars); raxm = min(rvars); raxr = raxM - raxm;
    rax = [raxm raxM] + 0.05*raxr*[-1 1]; raxt = linspace(rax(1),rax(2),5);
    raxt = unique(str2num(num2str(raxt,2))); rax = raxt([1 end]);	% Round numbers
    % Plot this figure
    set(figure(Figs(i)),'Position',[Figsp(i,:) Figss],...
        'Name',['Fourier spectrum statistics of ' Figsl{i}],...
        'Color','w','PaperPositionMode','auto');
    clf; hold on;
    % Plot each array
    [AXa,H1,H2] = plotyy(Xval,lambdaM,Xval,alphaM,'plot');
    set(AXa(1),'XLim',Xax,'XTick',[],'YLim',lax,'YTick',[]);
    set(AXa(2),'XLim',Xax,'XTick',[],'YLim',rax,'YTick',[]);
    set(H1,'LineStyle','-','Color',lcol)
    set(H2,'LineStyle','-','Color',rcol)
    [AXc,H5,H6] = plotyy(Xval,lambdam,Xval,NaN,'plot');
    set(AXc(1),'XLim',Xax,'XTick',XT,...
        'YLim',lax,'YTick',laxt,'Box','on');
    set(AXc(2),'XLim',Xax,'XTick',[],...
        'YLim',rax,'YTick',raxt);
    set(H5,'LineStyle','--','Color',lcol)
    hold off;
    set(Figs(i),'CurrentAxes',AXc(2)); Figsax = [Xax rax];
    set(get(AXc(2),'YLabel'),...
        'String',['\alpha_M' Figsunitsr{i}],'Color',rcol);
    set(AXc(2),'YColor',rcol);
    set(get(AXc(1),'XLabel'),'String','Iteration');
    set(get(AXc(1),'YLabel'),...
        'String',['\lambda_M, \lambda_m' Figsunitsl{i}],'Color',lcol);
    set(AXc(1),'YColor',lcol);
    text(XT,repmat(Figsax(4)+0.025*diff(Figsax(3:4)),size(XT)),...
        num2str(Ks(XT),3),'HorizontalAlignment','center','FontSize',6);
    text(mean(Figsax(1:2)),Figsax(4)+0.065*diff(Figsax(3:4)),'K');
    legend([H1 H5 H2],'\lambda_M','\lambda_m','\alpha_M','Location','best');
    drawnow;
    if ~isempty(ENdir)
        print(Figs(i),'-loose','-r150',['-d' EXT],...
            sprintf([ENdir,'/fig%03d.%s'],Figs(i),EXT));
    end
end

Figs = 142;
Figss = [560 420]*2;
Figsp = mean(Figsp);
if plotfig(142)
    % Turn each cell array in stats.fig into an array:
    lambdaM_OD = cell2mat(stats.fig(140).lambdaM(whichones));
    lambdam_OD = cell2mat(stats.fig(140).lambdam(whichones));
    lambdaM_OR = cell2mat(stats.fig(141).lambdaM(whichones));
    lambdam_OR = cell2mat(stats.fig(141).lambdam(whichones));
    % Compute axes (left-side and right-side)
    lvars = [lambdaM_OD lambdam_OD lambdaM_OR lambdam_OR];
    laxM = max(lvars); laxm = min(lvars); laxr = laxM - laxm;
    lax = [laxm laxM] + 0.05*laxr*[-1 1]; laxt = linspace(lax(1),lax(2),5);
    laxt = unique(str2num(num2str(laxt,2))); lax = laxt([1 end]);	% Round numbers
    % Plot this figure
    set(figure(Figs),'Position',[Figsp Figss],...
        'Name',['Fourier spectrum statistics of OD and ORt'],...
        'Color','w','PaperPositionMode','auto');
    clf; hold on;
    % Plot each array
    H1 = plot(Xval,lambdaM_OD,'k-');
    hold on;
    H2 = plot(Xval,lambdaM_OR,'b-');
    H3 = plot(Xval,lambdam_OD,'k--');
    H4 = plot(Xval,lambdam_OR,'b--');
    hold off;
    set(gca,'XLim',Xax,'XTick',XT,'YLim',lax,'YTick',laxt,'Box','on');
    xlabel('Iteration'); ylabel('Wavelength (pixels)'); Figsax = [Xax lax];
    text(XT,repmat(Figsax(4)+0.025*diff(Figsax(3:4)),size(XT)),...
        num2str(Ks(XT),3),'HorizontalAlignment','center','FontSize',6);
    text(mean(Figsax(1:2)),Figsax(4)+0.065*diff(Figsax(3:4)),'K');
    legend([H1 H2 H3 H4],'\lambda_M(OD)','\lambda_M(OR)',...
        '\lambda_m(OD)','\lambda_m(OR)','Location','best');
    drawnow;
    if ~isempty(ENdir)
        print(Figs,'-loose','-r150',['-d' EXT],...
            sprintf([ENdir,'/fig%03d.%s'],Figs,EXT));
    end
end


% --- Figures 150, 160: statistics for pinwheel in fig. 50 (pinw: numbers and
%     densities of pinwheels) and pinwheels-on-OD-borders (pinw: numbers and
%     densities of pinwheels).
Figs = [150 160];
Figsl = {'Pinwheel statistics','Pinwheels-on-OD-borders statistics'};
Figsunitsl = {'',''};
Figsunitsr = {'',''};
Figss = [560 420]*2;
Figsp = [1 1;1+Figss(1) 1];
lwhichones = length(whichones);
for i=find(plotfig(Figs))
    % Extract statistics from the "pinw" cell array in stats.fig(Figs):
    pinw = reshape(cell2mat(stats.fig(Figs(i)).pinw(whichones)),3,6,lwhichones);
    npinw = reshape(pinw(1,1,:),1,lwhichones);
    npinwp = reshape(pinw(2,1,:),1,lwhichones);
    npinwn = reshape(pinw(3,1,:),1,lwhichones);
    dpinw_ORlM = reshape(pinw(1,3,:),1,lwhichones);
    dpinw_ODlM = reshape(pinw(1,4,:),1,lwhichones);
    dpinw_ORlm = reshape(pinw(1,5,:),1,lwhichones);
    dpinw_ODlm = reshape(pinw(1,6,:),1,lwhichones);
    % Compute axes (left-side and right-side)
    lvars = [npinw npinwp npinwn]; lcol = 'k';
    laxM = max(lvars); laxm = min(lvars); laxr = laxM - laxm;
    lax = [laxm laxM] + 0.05*laxr*[-1 1]; lax(1) = max(0,lax(1));
    laxt = linspace(lax(1),lax(2),5);
    laxt = unique(str2num(num2str(laxt,2))); lax = laxt([1 end]);	% Round numbers
    rvars = [dpinw_ORlM dpinw_ODlM dpinw_ORlm dpinw_ODlm]; rcol = 'b';
    raxM = max(rvars); raxm = min(rvars); raxr = raxM - raxm;
    rax = [raxm raxM] + 0.05*raxr*[-1 1]; rax(1) = max(0,rax(1));
    raxt = linspace(rax(1),rax(2),5);
    raxt = unique(str2num(num2str(raxt,2))); rax = raxt([1 end]);	% Round numbers
    % Plot this figure
    set(figure(Figs(i)),'Position',[Figsp(i,:) Figss],...
        'Name',Figsl{i},...
        'Color','w','PaperPositionMode','auto');
    clf; hold on;
    % Plot each array
    [AXa,H1,H2] = plotyy(Xval,npinw,Xval,dpinw_ORlM,'plot');
    set(AXa(1),'XLim',Xax,'XTick',[],'YLim',lax,'YTick',[]);
    set(AXa(2),'XLim',Xax,'XTick',[],'YLim',rax,'YTick',[]);
    set(H1,'LineStyle','-','Color',lcol)
    set(H2,'LineStyle','-','Color',rcol)
    [AXb,H3,H4] = plotyy(Xval,npinwp,Xval,dpinw_ODlM,'plot');
    set(AXb(1),'XLim',Xax,'XTick',[],'YLim',lax,'YTick',[]);
    set(AXb(2),'XLim',Xax,'XTick',[],'YLim',rax,'YTick',[]);
    set(H3,'LineStyle','--','Color',lcol)
    set(H4,'LineStyle','--','Color',rcol)
    [AXc,H5,H6] = plotyy(Xval,npinwn,Xval,dpinw_ORlm,'plot');
    set(AXc(1),'XLim',Xax,'XTick',[],'YLim',lax,'YTick',[]);
    set(AXc(2),'XLim',Xax,'XTick',[],'YLim',rax,'YTick',[]);
    set(H5,'LineStyle',':','Color',lcol)
    set(H6,'LineStyle',':','Color',rcol)
    [AXd,H7,H8] = plotyy(Xval,NaN,Xval,dpinw_ODlm,'plot');
    set(AXd(1),'XLim',Xax,'XTick',XT,...
        'YLim',lax,'YTick',laxt,'Box','on');
    set(AXd(2),'XLim',Xax,'XTick',[],...
        'YLim',rax,'YTick',raxt);
    set(H8,'LineStyle','-.','Color',rcol)
    hold off;
    set(Figs(i),'CurrentAxes',AXd(2)); Figsax = [Xax rax];
    set(get(AXd(2),'YLabel'),...
        'String',['\rho_{OR,M}, \rho_{OD,M}, \rho_{OR,m}, \rho_{OD,m}'...
        Figsunitsr{i}],'Color',rcol);
    set(AXd(2),'YColor',rcol);
    set(get(AXd(1),'XLabel'),'String','Iteration');
    set(get(AXd(1),'YLabel'),'String',...
        ['# pinwheels (all, +, -)' Figsunitsl{i}],'Color',lcol);
    set(AXd(1),'YColor',lcol);
    text(XT,repmat(Figsax(4)+0.025*diff(Figsax(3:4)),size(XT)),...
        num2str(Ks(XT),3),'HorizontalAlignment','center','FontSize',6);
    text(mean(Figsax(1:2)),Figsax(4)+0.065*diff(Figsax(3:4)),'K');
    legend([H1 H3 H5 H2 H4 H6 H8],...
        '# pinwheels (all)','# pinwheels (+)','# pinwheels (-)',...
        '\rho_{OR,M}','\rho_{OD,M}','\rho_{OR,m}','\rho_{OD,m}','Location','best');
    drawnow;
    if ~isempty(ENdir)
        print(Figs(i),'-loose','-r150',['-d' EXT],...
            sprintf([ENdir,'/fig%03d.%s'],Figs(i),EXT));
    end
end

% --- Figures 170-171: correlation statistics for figs. 70-75 (C,c).
Figs = 170:171;
Figsl = {' (Pearson''s correlation)',' (angle cosine)'};
Figsl2 = {'r','c'};
Figsunitsl = {'',''};
Figsunitsr = {'',''};
Figss = [560 420]*2;
Figsp = [1 1;1+Figss(1) 1];
for i=find(plotfig(Figs))
    % Turn each cell array in stats.fig into an array:
    Cc = [];
    for j=whichones
        switch i
            case 1
                C = stats.fig(Figs(i)).C{j};
            case 2
                C = stats.fig(Figs(i)).c{j};
        end
        Cc = [Cc;C(find(tril(C,-1)))'];
    end
    % Compute axes
    lax = [-1.1 1.1];
    laxt = linspace(lax(1),lax(2),5);
    laxt = unique(str2num(num2str(laxt,2))); lax = laxt([1 end]);	% Round numbers
    % Plot this figure
    set(figure(Figs(i)),...
        'Position',[Figsp(i,:) Figss],...
        'Name',['Correlation statistics for gradients' Figsl{i}],...
        'Color','w','PaperPositionMode','auto');
    plot(Xval,Cc);
    Figsax = [Xax lax]; axis(Figsax); set(gca,'XTick',XT,'Box','on');
    xlabel('Iteration'); ylabel(Figsl2{i});
    text(XT,repmat(Figsax(4)+0.025*diff(Figsax(3:4)),size(XT)),...
        num2str(Ks(XT),3),'HorizontalAlignment','center','FontSize',6);
    text(mean(Figsax(1:2)),Figsax(4)+0.065*diff(Figsax(3:4)),'K');
    legend('gVF-gOD','gVF-gORt','gVF-ORr','gOD-gORt','gOD-ORr','gORt-ORr','Location','best');
    drawnow;
    if ~isempty(ENdir)
        print(Figs(i),'-loose','-r150',['-d' EXT],...
            sprintf([ENdir,'/fig%03d.%s'],Figs(i),EXT));
    end
end
% --------------------------------------------------------------------------
