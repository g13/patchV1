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
ENtraining = 'canonical';
% ----------------------------- USER VALUES -----------------------------
gen = 'twister';
clear seed
seed = 1425269650;
%seed = 1422230161;
if exist('seed','var')
    rng(seed,gen);
else
    scurr = rng('shuffle')
    seed = scurr.Seed;
end

% Processing of intermediate (historical) parameters:
ENproc = 'save';		% One of 'var', 'save', 'varplot', 'saveplot'
%ENproc = 'varplot';		% One of 'var', 'save', 'varplot', 'saveplot'
%ENproc = 'saveplot';
ENfilename = ['rec_93'];   % Simulation name ***
non_cortical_lr = false;
% non_cortical_lr = true;
% cortical_shape = true;
cortical_shape = false;
uniform_LR = true;
% uniform_LR = false;
if ~cortical_shape
%     test_dw = 15;
%     test_dh = 20;
    test_dw = 5;
    test_dh = 7;
end
copyfile('parameters.m',[ENfilename,'_p.m']);
% Objective function weights

alpha = 1;			% Fitness term weight
beta = 100;  		% Tension term weight
% Training parameters
iters = 10;%21;			% No. of annealing iterations (saved)
max_it = 10;%12;			% No. of annealing iterations (not saved) ***
Kin = 0.14;			% Initial K ***
Kend = 0.02;        % Final K ***    
% - VFx: Nx points in [0,1], with interpoint separation dx.
Nx = 8;			% Number of points along VFx ***
nvf = 4;
rx = [0 0.16]*nvf;			% Range of VFx
dx = diff(rx)/(Nx-1);		% Separation between points along VFx
% - VFy: ditto for Ny, dy.
Ny = 13;			% Number of points along VFy ***
ry = [0 0.26]*nvf;			% Range of VFy
dy = diff(ry)/(Ny-1);		% Separation between points along VFy    
d = (dx + dy)/2;
% - OD: NOD values in range rOD, with interpoint separation dOD.
l = 0.12;			% Half-separation between OD layers ***
NOD = 2;			% Number of points along OD
rOD = [-l l];			% Range of OD
dOD = diff(rOD)/(NOD-1);	% Separation between points along OD
%  coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
%  r = 6*l/pi
r = 1.0*l;			% OR modulus -- the radius of pinwheel
NOR = 8;	 %8;		% Number of points along OR    
% for myCortex patch
ODnoise = l*0.0;
ODabsol = 1.0;
nG = 2;
G = [64 104]*nG;		% Number of centroids *** 
ecc = 2;
nod = 25;
a = 0.635; b = 96.7; k = sqrt(140)*0.873145;
fign = 106;

% Canonical training set: uniform grids for retinotopy, OD and OR as
% follows:
% -- T is created for ploting purposes
% -- retinotopy to be generated independently
%%%% copy from parameters.m

% - OR: NOR values in a periodic interval [-pi/2,pi/2] with modulus r,
rORt = [-pi/2 pi/2];		% Range of ORtheta
rORr = [0 r];			% Range of ORr
tmp1 = linspace(rORt(1),rORt(2),NOR+1); tmp1 = tmp1(1:NOR);
%dOR = diff(rORt)/(NOR-1)*r;
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

%     T = ENtrset('grid',zeros(1,3),...
%         linspace(rx(1),rx(2),Nx),...	% VFx
%         linspace(ry(1),ry(2),Ny),...	% VFy
%         linspace(rOD(1),rOD(2),NOD));

%linspace(rSF(1),rSF(2),NSF),... % SF
[N,D] = size(T);

% match feature with id
id = struct('VFx',1,'VFy',2,'OD',3,'ORx',4,'ORy',5,'OR',6,'ORr',7);
%     id = struct('VFx',1,'VFy',2,'OD',3,'ORx',4,'ORy',5,'OR',6,'ORr',7);
%     id = struct('VFx',1,'VFy',2,'OD',3);
    % Ranges of stimuli variables (for greyscales, etc.)
%     v = [1 rx;2 ry;3 rOD];
v = [1 rx;2 ry;3 rOD;4 -r r;5 -r r;6 -pi/2 pi/2;7 0 r];
%v = [1 NaN NaN;2 NaN NaN;3 -l l;4 NaN NaN;5 NaN NaN;6 -pi/2 pi/2;7 0 r];
%v = [1 rOD; 2 -r r; 3 -r r; 4 -pi/2 pi/2; 5 0 r]; % -- last 2 rows: OR augmented by polar
disp([num2str(size(T,1)),' references(cities) x ',num2str(size(T,2)),' features']);
if isfield(id, 'ORx')
    [tmp1,tmp2] = pol2cart(2*T(:,id.ORx),T(:,id.ORy)); % polar coords to Cartesian coords
    T(:,[id.ORx,id.ORy]) = [tmp1 tmp2];			% ORx, ORy
end

% The training set is slightly noisy to avoid symmetry artifacts.

%     T = T + (rand(size(T))-1)/10000;		% Tiny noise
    
% Training parameters
max_cyc = 1;		% Number of cycles per annealing iteration
min_K = eps;		% Smallest K before K is taken as 0
tol = -1;			% Smallest centroid update
method = 'Cholesky';		% Training method
annrate = (Kend/Kin)^(1/(max_it*iters-1));	% Annealing rate
disp(['annealing rate: ', num2str(annrate)]);
Ksched = Kin*repmat(annrate^max_it,1,iters).^(0:iters-1);
%     Ksched = Ksched(4:10);
%     iters = 7;

    % Elastic net configuration: 2D
%     G = [64 104]*nG;		% Number of centroids ***    
%     G = [64 96]*nG;		% Number of centroids ***   3.4
bc = 'nonperiodic';		% One of 'nonperiodic', 'periodic' ***
p = 1;			% Stencil order (of derivative) ***
s = {[0 -1 1],[0;-1;1]};	% Stencil list ***
L = length(G); M = prod(G);
% For non-rectangular cortex shapes, create a suitable Pi here:

resol = 1;
disp(['estimated lambda OD =', num2str(8*l/(1/G(1)+1/G(2)))]); %
disp(['estimated lambda OR =', num2str(2*pi*r/(1/G(1)+1/G(2)))]); %
if cortical_shape
    [Pi, W, LR] = myCortex(G,ecc,a,b,k,resol, nod, rOD*ODabsol, ODnoise, ~non_cortical_lr && ~uniform_LR, ~uniform_LR * fign);
else
    Pi = zeros(G);
    Pi(1+test_dw*nG:G(1)-nG*test_dw, 1+test_dh*nG:G(2)-nG*test_dh) = 1;
    %Pi = [];			% Don't disable any centroid
    W = G(1)-nG*test_dw;			% Net width along 1st var. (arbitrary units)
end
if non_cortical_lr || ~cortical_shape
    LR = ones(G(1),G(2));
    OD_width = 20; % in pixels
    for i=1:round(G(1)/OD_width)
        if mod(i,2) == 0
            is = (i-1)*OD_width+1;
            ie = min(is + OD_width-1, G(1));
            LR(is:ie, :) = -1;
        end
    end
    LR = LR*l*ODabsol + ODnoise*randn(G(1),G(2));
    LR(LR < -l) = -l;
    LR(LR > l) = l;
end
%     pause;
%gridVF = zeros(G(1), G(2),2);
%for iy =1:G(2)
%    gridVF(logical(Pi(:,iy)),iy,1) = linspace(rx(1),rx(2),sum(Pi(:,iy)))';
%end
%for ix =1:G(1)
%    gridVF(ix,logical(Pi(ix,:)),2) = linspace(ry(1),ry(2),sum(Pi(ix,:)));
%end
Pi = Pi(:)';
[S,DD,knot,A,LL] = ENgridtopo(G,bc,Pi,s{:});
normcte = ENstennorm(G,W,p,s{:});	% Normalisation constant
% $$$   % Use normcte = 1; when disregarding the step size and resolution:
% $$$   normcte = 1;

% Initial elastic net: retinotopic with some noise and random, uniform
% OD and OR.

mu = ENtrset('grid',zeros(1,2),...		% Small noise
    linspace(rx(1),rx(2),G(1)),...	% VFx
    linspace(ry(1),ry(2),G(2)));	% VFy
%    mu = reshape(gridVF, M, 2);
if uniform_LR
    mu = [mu, ENtrset('uniform',rOD,M)];
else
    mu = [mu reshape(LR,M,1)];		% OD
end

%     mu = [mu ENtrset('uniform',...
%         [-pi/2,pi/2;...		% ORtheta
%         0,r],...        %ORr
%         M)];

%     mu = ENtrset('grid',zeros(1,2),...		% Small noise
%        linspace(rx(1),rx(2),G(1)),...	% VFx
%        linspace(ry(1),ry(2),G(2)));	% VFy
%     mu = [mu ENtrset('uniform',[-l l;...		% OD
mu = [mu ENtrset('uniform',[-pi/2 pi/2;...		% ORtheta
   0 r],... % ORr
   M)];
%         -lSF lSF],...   % -- SF
%         M)];
if isfield(id, 'ORx')    
    [tmp1,tmp2] = pol2cart(2*mu(:,id.ORx),mu(:,id.ORy));
    mu(:,[id.ORx, id.ORy]) = [tmp1 tmp2];			% ORx, ORy
end
if ~isempty(Pi), mu(Pi==0,:) = 0; end;
disp([num2str(size(mu,1)),' centroids x ',num2str(size(mu,2)),' features']);

betanorm = beta*normcte;	% Normalised beta


% Actual ranges of centroids (for axes)
if isfield(id, 'OR')
    [tmp1,tmp2] = cart2pol(mu(:,id.ORx),mu(:,id.ORy));
    tmp1 = tmp1 / 2;
    murange = [mu tmp1 tmp2];
else
    murange = mu;
end
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
            myV1replay(G,bc,ENlist,v,1,T,Pi,murange,id);
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
            myV1replay(G,bc,ENlist,v,1,T,Pi,murange,id);
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
%     disp(['K#',num2str(ENcounter),' = ', num2str(Ksched(ENcounter))]);
    [mu,stats] = ENtr_ann(Tr,S,Pi,mu,Ksched(ENcounter),alpha,betanorm,...
        annrate,max_it,max_cyc,min_K,tol,method);
    if isfield(id, 'OR')
        [tmp1,tmp2] = cart2pol(mu(:,id.ORx),mu(:,id.ORy));
        tmp1 = tmp1 / 2;
        murange = [mu tmp1 tmp2;murange];
    else
        murange = mu;
    end
    murange = [min(murange);max(murange)];
    
    % Process parameters:
    switch ENproc
        case {'var','varplot'}
            ENlist(ENcounter+1).mu = mu;
            ENlist(ENcounter+1).stats = stats;
            if strcmp(ENproc,'varplot')
                myV1replay(G,bc,ENlist,v,ENcounter+1,T,Pi,murange,id);
            end
        case {'save','saveplot'}
            save(sprintf('%s%04d.mat',ENfilename,ENcounter),'mu','stats','murange');
            if strcmp(ENproc,'saveplot')
                myV1replay(G,bc,struct('mu',mu,'stats',stats),v,1,T,Pi,murange,id);
            end
        otherwise
            % Do nothing.
    end
    disp(['K=',num2str(Ksched(ENcounter))]);
end

% Save results
switch ENproc
    case {'var','varplot'}
        eval(['save ' ENfilename '.mat ENfilename '...
            'Nx rx dx Ny ry dy NOD rOD dOD l NOR rORt rORr r seed v ' ...
            'ENlist murange id ' ...
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
            'ENlist murange id ' ...
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
        myV1replay(G,bc,ENlist,v,1,T,Pi,murange,id,[],[],[100, 102]);
    otherwise
        % Do nothing.
end

disp(['ratio of L/R = ', num2str(sum(mu(logical(Pi),id.OD) < 0)/sum(mu(logical(Pi),id.OD) > 0))]);
myV1stats(G,bc,ENlist,v,'last',T,Pi,murange,id,[],[],[1,2,3,4,7,100,102,34, 40, 41, 50, 54, 60]);
