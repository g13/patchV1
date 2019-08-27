clear all
clc;
addpath('/home/wd554/MATLAB/ElasticNet/')
 
gen = 'twister';
clear seed
seed = 1425269650;
%seed = 1422230161;
format = '-dpng';

plots = true;
new = true;
ENproc = 'var';		%2One of 'var', 'save', 'varplot', 'saveplot'
% Processing of intermediate (historical) parameters:
var = 'beta';
ENfilename0 = ['cortex_nG4-',var];   % Simulation name ***
range = [2,4,6,8,10];
%range = [2];
%non_cortical_lr = false;
non_cortical_lr = true;
% cortical_shape = false;
cortical_shape = true;
uniform_LR = true;
%uniform_LR = false;

mkdir(ENfilename0);
copyfile('parameters.m',[ENfilename0,'/',ENfilename0,'_p.m']);
poolobj = gcp('nocreate'); % If no pool, do not create new one.
if ~isempty(poolobj)
    if ~ (poolobj.NumWorkers == length(range))
        delete(poolobj);
        parpool(length(range));
    end
end
if exist('seed','var')
    rng(seed,gen);
else
    scurr = rng('shuffle')
    seed = scurr.Seed;
end
parfor i = 1:length(range)
%for i = 1:length(range)
    % for non_cortical_shape edge boundaries
    test_dw = 5;
    test_dh = 7;
    % Objective function weights

    alpha = 1;		% Fitness term weight
	beta = 200*range(i)/range(end);
    % Training parameters
    iters = 10;%21;			% No. of annealing rates (saved)
    max_it = 5;%12;			% No. of iterations per annealing rate (not saved) ***
    Kin = 0.15;			% Initial K ***
    Kend = 0.02;        % Final K ***    
    % - VFx: Nx points in [0,1], with interpoint separation dx.
    Nx = 8;			% Number of points along VFx ***
	nvf = 8;
    %nvf = range(i);
    rx = [0 0.16]*nvf;			% Range of VFx
    dx = diff(rx)/(Nx-1);		% Separation between points along VFx
    % - VFy: ditto for Ny, dy.
    Ny = 12;			% Number of points along VFy ***
    ry = [0 0.24]*nvf;			% Range of VFy
    dy = diff(ry)/(Ny-1);		% Separation between points along VFy    
    d = (dx + dy)/2;
    % - OD: NOD values in range rOD, with interpoint separation dOD.
    l = 0.12;			% Half-separation between OD layers ***
    NOD = 2;			% Number of points along OD
    rOD = [-l l];			% Range of OD
    dOD = diff(rOD)/(NOD-1);	% Separation between points along OD
    %  coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
    %  r = 6*l/pi
	%r = (1.0+range(i)/range(end))*l;			% OR modulus -- the radius of pinwheel
	r = 1.4*l;
    NOR = 8;	 %8;		% Number of points along OR    
    % for myCortex patch
    ODnoise = l*0.0;
    ODabsol = 1.0;
    nG = 4;
    G = round([64 104]*nG);		% Number of centroids *** 
    ecc = 2;
    nod = 25;
    a = 0.635; b = 96.7; k = sqrt(140)*0.873145;
    fign = 106;
    ENfilename = [var,'-',num2str(range(i))];
	if i == 1
		saveLR = true;
	else
		saveLR = false;
	end
    stats(i) = myV1driver(seed,ENproc,ENfilename0,ENfilename,non_cortical_lr,cortical_shape,uniform_LR,test_dw,test_dh,alpha,beta,iters,max_it,Kin,Kend,Nx,nvf,rx,dx,Ny,ry,dy,l,NOD,rOD,dOD,r,NOR,ODnoise,ODabsol,nG,G,ecc,nod,a,b,k,fign,plots,new,saveLR);
end
nnpinw = [stats.npinw];
nnpinw = nnpinw./mean(nnpinw);
normD_overMean = [stats.normD_overMean];
OD_OR_I_ang_overMean = [stats.OD_OR_I_ang_overMean];
OD_OR_B_ang_overMean = [stats.OD_OR_B_ang_overMean];
OD_B_ang_overMean = [stats.OD_B_ang_overMean];
OR_B_ang_overMean = [stats.OR_B_ang_overMean];

normD_mode = [stats.normD_mode];
OD_OR_I_angMode = [stats.OD_OR_I_angMode];
OD_OR_B_angMode = [stats.OD_OR_B_angMode];
OD_B_angMode = [stats.OD_B_angMode];
OR_B_angMode = [stats.OR_B_angMode];
h = figure;
subplot(1,2,1);
hold on
plot(range,nnpinw,'or');
plot(range,normD_overMean,'-k');
plot(range,normD_mode,':k');
plot(range,OD_OR_I_ang_overMean,'s-');
plot(range,OD_OR_B_ang_overMean,'*-');
plot(range,OD_B_ang_overMean,'^-');
plot(range,OR_B_ang_overMean,'>-');
plot(range,zeros(size(range))+0.5,':g');
legend({'npinw','normDamp','normDmode','ODOR_I','ODOR_B','OD_B','OR_B'});
subplot(1,2,2);
hold on
plot(range,OD_OR_I_angMode,'s-');
plot(range,OD_OR_B_angMode,'*-');
plot(range,OD_B_angMode,'^-');
plot(range,OR_B_angMode,'>-');
legend({'ODOR_I','ODOR_B','OD_B','OR_B'});
figname = [ENfilename0,'/',ENfilename0,'-stats'];

set(gcf,'PaperPositionMode','auto')
print(gcf, figname, format, '-r150');
saveas(gcf,[figname,'.fig']);

eval(['save ', ENfilename0 '/' ENfilename0 '-stats.mat stats']); 
