clear all
clc;
addpath(genpath('/home/wd554/MATLAB/'))
 
gen = 'twister';
clear seed
seed = 1435259639;
%seed = 1422230161;
format = '-dpng';

separateData = true; % set to false for ease of comparison in one folder
plots = true;
new = true;
ENproc = 'save';		%2One of 'var', 'save', 'varplot', 'saveplot'
% Processing of intermediate (historical) parameters:
name = 'uniformXY-hb1';
var = 'ODl'
equi = 'cortex';
old = true;
heteroAlpha = 1; % -1 reciprocal, 0 identity, 1 area
switch heteroAlpha
case -1
	wtString = 'r';
case 0
	wtString = 'i';
case 1
	wtString = 'a';
end
VFpath = 'Training_pos-hb-1.bin';
if isempty(VFpath)
	ENfilename0 = [name,'-',equi,'-',wtString,'-',var]   % Simulation name ***
else
	ENfilename0 = [name,'-ext-',var]   % Simulation name ***
end
plotting = 'all' % 'all', 'first', >0 frame, <0 frame:end
%range = [6,8,10,12,14];
%range = [1.2,1.3,1.4,1.5,1.6];
%range = [10,15,20,25,30];
%range = [1.0,1.25,1.5,1.75,2.0];
%range = [5,7.5,10,12.5,15];
range = [0.8,0.9,1.0,1.1,1.2];
%range = [1.0];
%range = [1];
cortical_VF = 'cortex';
non_cortical_LR = false;
% non_cortical_LR = true;
uniform_LR = true;
%equi = 'VF';
weightType = 'min';
saveLR = true;
%weightType = 'area';
%uniform_LR = false;
% SET both LR to false for manual_LR
% cortical_shape = false;
cortical_shape = true;
if exist(ENfilename0, 'dir') && new
    rmdir(ENfilename0,'s');
end
if ~exist(ENfilename0, 'dir')
    mkdir(ENfilename0);
end

copyfile([mfilename,'.m'],[ENfilename0,'/',ENfilename0,'_p.m']);
if length(range) > 1
%if false
	poolobj = gcp('nocreate'); % If no pool, do not create new one.
	if ~isempty(poolobj)
	    if ~ (poolobj.NumWorkers == length(range))
	        delete(poolobj);
	        parpool(length(range));
	    end
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

    alpha = 1.0;		% Fitness term weight
	%beta = 50*range(i);
	beta = 500;
    % Training parameters
	iters = 10; %21;			% No. of annealing rates (saved)
    max_it = 10;		        % No. of iterations per annealing rate (not saved) ***
	%max_it = round(20 *range(i)/range(end)); 
    Kin = 0.15;			% Initial K ***
    Kend = 0.03;        % Final K ***    
    % - VFx: Nx points in [0,1], with interpoint separation dx.
	%Nx = round(17*range(i))			% Number of points along VFx ***
	Nx = 17;
	%Nx = range(i)
	%Nx = round(16*range(i)); %			% Number of points along VFx ***
	%nvf = range(i);
    nvf = 10;
    rx = [0 0.25]*nvf;			% Range of VFx
    % - VFy: ditto for Ny, dy.
	%
	% even
	%Ny = round(34*range(i)) 
	Ny = 34;
	%Ny = 60 - range(i);
	if mod(Nx,2) == 1,Nx = Nx + 1; end
	if mod(Ny,2) == 1,Ny = Ny - 1; end
	%if cortical_VF
	%	assert(Nx*2 == Ny);
	%end
	%Ny = round(25*range(i));			% Number of points along VFy ***
    ry = [0 0.5]*nvf;			% Range of VFy
    % - OD: NOD values in range rOD, with interpoint separation dOD.
	l = 0.11*range(i);
    NOD = 2;			% Number of points along OD
    rOD = [-l l];			% Range of OD
    %  coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
    %  r = 6*l/pi
	%r = range(i)*l;			% OR modulus -- the radius of pinwheel
	r = 1.4*l;
    NOR = 8;	 %8;		% Number of points along OR    
    % for myCortex patch
    ODnoise = l*0.0;
    ODabsol = 1.0;
    nG = 2;
    G = round([64 104]*nG);		% Number of centroids *** 
    ecc = 2;
    nod = 25;
    a = 0.635; b = 96.7; k = sqrt(140)*0.873145;
    fign = 106;
    ENfilename = [var,'-',num2str(range(i))];
    stats(i) = myV1driver(seed,ENproc,ENfilename0,ENfilename,non_cortical_LR,cortical_VF,cortical_shape,uniform_LR,test_dw,test_dh,alpha,beta,iters,max_it,Kin,Kend,Nx,nvf,rx,Ny,ry,l,NOD,rOD,r,NOR,ODnoise,ODabsol,nG,G,ecc,nod,a,b,k,i,plots,new,saveLR,separateData,plotting,heteroAlpha,equi,weightType,VFpath,old);
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
