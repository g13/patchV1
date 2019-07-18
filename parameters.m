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
ENfilename = ['rec_93'];
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
    %   coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
%     r = 6*l/pi
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
    % Simulation name ***
