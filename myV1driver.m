% Driver file for an EN simulation for cortical maps:
% 2D cortex in stimulus space (VFx,VFy,OD,ORx,ORy) where VF = visual
% field, OD = ocular dominance and OR = orientation.
%
% Note: the net is considered nonperiodic. For a periodic net, the
% starting centroids and the VF variables of the training set must be
% wrapped; thus, every VF variable must be coded using two variables.
%
% To set up the parameters (training set, initial centroids, net
% configuration and training parameters), edit the lines enclosed by
% "USER VALUES"; the most likely values to be edited are marked "***".
% Alternatively, if the variable ENsetupdone is set to 'yes', no
% parameters are set up; this is useful to use previously set parameters
% (e.g. read from a file or set manually in Matlab).

% ENtraining determines the training set Tr that is used at each iteration
% in the ENcounter loop. It can be one of these:
% - 'canonical': use T, i.e., a uniform grid in (VFx,VFy,OD,ORt).
% - 'noisy': add uniform noise to the canonical T at each iteration.
% - 'uniform1': generate a uniform sample in the (rectangular) domain of
%   (VFx,VFy,OD,ORt,ORr) at each iteration.
% - 'uniform2': like uniform1, but force OD to be binary in {-l,l}
%   and ORr = r.
% The canonical training set T is created independently of the value of
% ENtraining and saved in ENfilename. It is used for plotting, as a
% scaffolding for the underlying continuous set of training vectors.
% 'noisy' and 'uniform*' approximate online training over the continuous
% domain of (VFx,VFy,OD,ORt).

addpath('C:\Users\gueux\Dropbox\Visual\Macaque V1\Model\Carreira_et-al-2005')
ENtraining = 'canonical';

if exist('ENsetupdone') ~= 1 ENsetupdone = 'no'; end;
if ~strcmp(ENsetupdone,'yes')
    % ----------------------------- USER VALUES -----------------------------
    % Processing of intermediate (historical) parameters:
    %ENproc = 'varplot';		% One of 'var', 'save', 'varplot', 'saveplot'
    ENproc = 'saveplot';
    
    seed = 317;			% Seed for random number generator
    rand('state',seed);
    
    % Canonical training set: uniform grids for retinotopy, OD and OR as
    % follows:
    % -- T is created for ploting purposes
    % -- retinotopy to be generated independently
    
    % - VFx: Nx points in [0,1], with interpoint separation dx.
    Nx = 10;			% Number of points along VFx ***
    rx = [0 1];			% Range of VFx
    dx = diff(rx)/(Nx-1);		% Separation between points along VFx
    % - VFy: ditto for Ny, dy.
    Ny = 10;			% Number of points along VFy ***
    ry = [0 1];			% Range of VFy
    dy = diff(ry)/(Ny-1);		% Separation between points along VFy
    
    % - OD: NOD values in range rOD, with interpoint separation dOD.
    l = 0.14;			% Half-separation between OD layers ***
    NOD = 2;			% Number of points along OD
    rOD = [-l l];			% Range of OD
    dOD = diff(rOD)/(NOD-1);	% Separation between points along OD
    % - OR: NOR values in a periodic interval [-pi/2,pi/2] with modulus r,
    %   coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
    NOR = 12;			% Number of points along OR
    r = 0.2;			% OR modulus *** -- seems to mean the radius of pinwheel
    rORt = [-pi/2 pi/2];		% Range of ORtheta
    rORr = [0 r];			% Range of ORr
    tmp1 = linspace(rORt(1),rORt(2),NOR+1); tmp1 = tmp1(1:end-1);
    % - SF: spatial frequency, same as OD, to be tested with larger NSF
    lSF = 0.14;
    NSF = 2;
    rSF = [-lSF lSF];
    dSF = diff(rSF)/(NSF-1);
    
    T = ENtrset('grid',zeros(1,5),...
        linspace(rx(1),rx(2),Nx),...	% VFx
        linspace(ry(1),ry(2),Ny),...	% VFy
        linspace(rOD(1),rOD(2),NOD),...	% OD
        tmp1,...				% ORtheta
        linspace(rORr(1),rORr(2),1));%,... % ORr
        %linspace(rSF(1),rSF(2)));
    [tmp1,tmp2] = pol2cart(2*T(:,4),T(:,5)); % polar coords to Cartesian coords
    T(:,4:5) = [tmp1 tmp2];			% ORx, ORy
    % The training set is slightly noisy to avoid symmetry artifacts.
    T = T + (rand(size(T))-1)/10000;		% Tiny noise
    [N,D] = size(T);
    
    % Elastic net configuration: 2D
    G = [70 110];		% Number of centroids ***
    bc = 'nonperiodic';		% One of 'nonperiodic', 'periodic' ***
    p = 1;			% Stencil order (of derivative) ***
    s = {[0 -1 1],[0;-1;1]};	% Stencil list ***
    %W = 100;			% Net width along 1st var. (arbitrary units)
    L = length(G); M = prod(G);
    % For non-rectangular cortex shapes, create a suitable Pi here:
    %Pi = [];			% Don't disable any centroid
    ecc = 3.4; 
    a = 1.09; b = 90; k = 19.3;
    resol = 10;
    [Pi, W] = myCortex(G,ecc,a,b,k,resol);
    Pi = Pi(:);
    [S,DD,knot,A,LL] = ENgridtopo(G,bc,Pi,s{:});
    normcte = ENstennorm(G,W,p,s{:});	% Normalisation constant
    % $$$   % Use normcte = 1; when disregarding the step size and resolution:
    % $$$   normcte = 1;
    
    % Initial elastic net: retinotopic with some noise and random, uniform
    % OD and OR.
    mu = ENtrset('grid',zeros(1,2),...		% Small noise
        linspace(rx(1),rx(2),G(1)),...	% VFx
        linspace(ry(1),ry(2),G(2)));	% VFy
    mu = [mu ENtrset('uniform',[-l l;...		% OD
        -pi/2 pi/2;...		% ORtheta
        0 r],... % ORr
        size(mu,1))];	
        %-lSF lSF],...   % -- SF
        %size(mu,1))];	
    [tmp1,tmp2] = pol2cart(2*mu(:,4),mu(:,5));
    mu(:,4:5) = [tmp1 tmp2];			% ORx, ORy
    disp(size(mu));
    if ~isempty(Pi) mu(Pi==0,:) = 0; end;
    
    % Objective function weights
    alpha = 1;			% Fitness term weight
    beta = 10;			% Tension term weight
    betanorm = beta*normcte;	% Normalised beta
    
    % Training parameters
    iters = 10;			% No. of annealing iterations (saved)
    max_it = 5;			% No. of annealing iterations (not saved) ***
    max_cyc = 1;			% Number of cycles per annealing iteration
    min_K = eps;			% Smallest K before K is taken as 0
    tol = -1;			% Smallest centroid update
    method = 'Cholesky';		% Training method
    Kin = 0.2;			% Initial K ***
    Kend = 0.06;			% Final K ***
    annrate = (Kend/Kin)^(1/(max_it*iters-1));	% Annealing rate
    Ksched = Kin*[repmat(annrate^max_it,1,iters).^(0:iters-1)];
    
    % Simulation name ***
    ENfilename = ['s' num2str(p) 'b' num2str(beta)];
    % ----------------------------- USER VALUES -----------------------------
end

% Ranges of stimuli variables (for greyscales, etc.)
v = [1 rx;2 ry;3 rOD;4 -r r;5 -r r;6 -pi/2 pi/2;7 0 r];
%v = [1 NaN NaN;2 NaN NaN;3 -l l;4 NaN NaN;5 NaN NaN;6 -pi/2 pi/2;7 0 r];
% Actual ranges of centroids (for axes)
[tmp1,tmp2] = cart2pol(mu(:,4),mu(:,5));
tmp1 = tmp1 / 2;
murange = [mu tmp1 tmp2];
murange = [min(murange);max(murange)];

% Initialisation
switch ENproc
    case {'var','varplot'}
        % Store parameters in a single variable called ENlist, which is
        % eventually saved in a file (whose name is contained in ENfilename).
        % This is useful for small nets.
        % ENlist(i) contains {mu,stats}, for i = 1, 2, etc.
        ENlist = struct('mu',mu,'stats',struct(...
            'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
            'cpu','','code',1,'it',0));
        % ENlist(1) contains only the initial value of mu.
        % I would prefer to have the initial value of mu in ENlist(0),
        % but crap Matlab does not allow negative or zero indices.
        if strcmp(ENproc,'varplot')
            ENV1replay2(G,bc,ENlist,v,1,T,Pi,murange);
        end
    case {'save','saveplot'}
        % Save parameters in separate files with names xxx0001, xxx0002, etc.
        % (assuming ENfilename = 'xxx'). File xxx0000 contains the simulation
        % parameters (including the initial value of mu). Once the training
        % loop finishes, all files are collected into a single variable. Thus,
        % the final result is the same as with 'var'.
        % This is useful when memory is scarce, and also more robust, since the
        % learning loop may be resumed from the last file saved in case of a
        % crash.
        eval(['save ' ENfilename '0000.mat ENfilename '...
            'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
            'N D L M G bc p s T Pi S DD knot A LL mu Kin Kend iters Ksched '...
            'alpha beta annrate max_it max_cyc min_K tol method '...
            'W normcte betanorm']);
        if strcmp(ENproc,'saveplot')
            ENlist = struct('mu',mu,'stats',struct(...
                'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
                'cpu','','code',1,'it',0));
            ENV1replay2(G,bc,ENlist,v,1,T,Pi,murange);
        end
    otherwise
        % Do nothing.
end

if exist('ENtr_annW')
    disp('Will use ENtr_ann2');
else
    disp('Will use ENtr_ann1');
end
Tr = T;
% Learning loop with processing of intermediate (historical) parameters:
for ENcounter = 1:length(Ksched)
    
    % Generate training set for this iteration. For 'uniform*' we use the
    % same N as for the canonical T.
    switch ENtraining
        case 'noisy'
            % Noise level to add to T as a fraction of the smallest variable range
            Tr_noise = 0.5;
            Tr = T + (rand(size(T))-0.5)*Tr_noise*min(diff([rx;ry;rOD;rORr]'));
        case 'uniform1'
            Tr = ENtrset('uniform',[rx;ry;rOD;rORt;rORr],N);
            [tmp1,tmp2] = pol2cart(2*Tr(:,4),Tr(:,5));
            Tr(:,4:5) = [tmp1 tmp2];			% ORx, ORy
        case 'uniform2'
            Tr = [ENtrset('uniform',[rx;ry;-l -l;rORt;r r],floor(N/2));...
                ENtrset('uniform',[rx;ry;l l;rORt;r r],N-floor(N/2))];
            [tmp1,tmp2] = pol2cart(2*Tr(:,4),Tr(:,5));
            Tr(:,4:5) = [tmp1 tmp2];			% ORx, ORy
        otherwise
            % Do nothing.
    end
    
    % Update parameters:
    [mu,stats] = ENtr_ann(Tr,S,Pi,mu,Ksched(ENcounter),alpha,betanorm,...
        annrate,max_it,max_cyc,min_K,tol,method);
    [tmp1,tmp2] = cart2pol(mu(:,4),mu(:,5));
    tmp1 = tmp1 / 2;
    murange = [mu tmp1 tmp2;murange];
    murange = [min(murange);max(murange)];
    
    % Process parameters:
    switch ENproc
        case {'var','varplot'}
            ENlist(ENcounter+1).mu = mu;
            ENlist(ENcounter+1).stats = stats;
            if strcmp(ENproc,'varplot')
                ENV1replay2(G,bc,ENlist,v,ENcounter+1,T,Pi,murange);
            end
        case {'save','saveplot'}
            save(sprintf('%s%04d.mat',ENfilename,ENcounter),'mu','stats','murange');
            if strcmp(ENproc,'saveplot')
                ENV1replay2(G,bc,struct('mu',mu,'stats',stats),v,1,T,Pi,murange);
            end
        otherwise
            % Do nothing.
    end
end

% Save results
switch ENproc
    case {'var','varplot'}
        eval(['save ' ENfilename '.mat ENfilename '...
            'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
            'ENlist murange ' ...
            'N D L M G bc p s T Pi S DD knot A LL Kin Kend iters Ksched '...
            'alpha beta annrate max_it max_cyc min_K tol method '...
            'W normcte betanorm']);
    case {'save','saveplot'}
        % Collect all files into a single one:
        load(sprintf('%s%04d.mat',ENfilename,0));
        ENlist = struct('mu',mu,'stats',struct(...
            'K',NaN,'E',[NaN NaN NaN],'time',[NaN NaN NaN],...
            'cpu','','code',1,'it',0));
        for ENcounter = 1:length(Ksched)
            load(sprintf('%s%04d.mat',ENfilename,ENcounter));
            ENlist(ENcounter+1).mu = mu;
            ENlist(ENcounter+1).stats = stats;
        end
        eval(['save ' ENfilename '.mat ENfilename '...
            'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
            'ENlist murange ' ...
            'N D L M G bc p s T Pi S DD knot A LL Kin Kend iters Ksched '...
            'alpha beta annrate max_it max_cyc min_K tol method '...
            'W normcte betanorm']);
        unix(['rm ' ENfilename '????.mat']);
    otherwise
        % Do nothing.
end

% Plot some statistics for the objective function value and computation time
switch ENproc
    case {'varplot','saveplot'}
        ENV1replay2(G,bc,ENlist,v,1,T,Pi,murange,DD,[],[11 12 100 101]);
    otherwise
        % Do nothing.
end

