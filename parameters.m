clear all
%clc;
%addpath(genpath('/home/wd554/MATLAB/'))
addpath(genpath('C:\Users\gueux\MATLAB\'))
 
gen = 'twister';
clear seed
seed = 1435269651;
%seed = 1422230161;
format = '-dpng';

exchange_nm = false;

separateData = true; % set to false for ease of comparison in one folder
plots = true;
new = false;
ENproc = 'save';		%2One of 'var', 'save', 'varplot', 'saveplot'
% Processing of intermediate (historical) parameters:
name = 'recap';
var = 'beta'
equi = 'VF';
cortical_VF = 'cortex';
old = false;
testRun = false;
randSeed = true;
heteroAlpha = 0; % -1 reciprocal, 0 identity, 1 area
switch heteroAlpha
case -1
	wtString = 'r';
case 0
	wtString = 'i';
case 1
	wtString = 'a';
end
ecc = 2.0;
%VFpath = '';
%nvfx = 2.6880
nvfy = 1.0
nvfx = 1.0

G0 = [224, 224];
rotate = [0,0,0,0];
nvfRange = [1.0, 1.0, 1.0, 1.0]*0.1;
xRange = [1.0, 1.0, 1.0, 1.0]*0.5;
yRange = [1.0, 1.0, 1.0, 1.0]*0.5;
rRange = [0.1, 0.16, 0.2, 0.24]*1;
lRange = [0.1, 0.12, 0.15, 0.18]*1;

nGrange = [1.0, 1.0, 1.0, 1.0];
betaRange = [10, 10, 10, 10]*40;
aRrange = [1.0, 1.0, 1.0, 1.0;
		   1.0, 1.0, 1.0, 1.0];

Kin = 0.15;			% Initial K ***
Kend = 0.02;        % Final K ***    
if testRun
	iters = 1; %21;			% No. of annealing rates (saved)
	max_it = 2;		        % No. of iterations per annealing rate (not saved) ***
else
	iters = 10; %21;			% No. of annealing rates (saved)
	max_it = 10;		        % No. of iterations per annealing rate (not saved) ***
end
%max_it = round(20 *range(i)/range(end)); 
figlist = [1,2,4,5,6,34,50,60,100,102];
%figlist = [1,2,4,5,6,15,16,34,50,60,100,102,600];
%VFpath = '/scratch/wd554/patchV1/or-ft10.bin';
VFpath = '';
if isempty(VFpath)
	ENfilename0 = [name,'-',equi,'-',wtString,'-',var]   % Simulation name ***
else
	ENfilename0 = [name,'-ext-',var]   % Simulation name ***
end
%plotting = 'all' % 'all', 'first', 'last', >0 frame, <0 frame:end
if testRun
	plotting = 'first'
else
	if new
		plotting = 'last'
	else
		plotting = 'all'
	end
end
if ~testRun
	range = 1:length(nGrange);
	%range = [1];
else
	range = [1];
end
non_cortical_LR = false;
% non_cortical_LR = true;
uniform_LR = true;
%equi = 'VF';
weightType = 'min';
saveLR = true;
%weightType = 'area';
%uniform_LR = false;
% SET both LR to false for manual_LR
cortical_shape = false;
%cortical_shape = true;

if length(range) > 1
%if false
	poolobj = gcp('nocreate'); % If no pool, do not create new one.
	if ~isempty(poolobj)
	    if ~ (poolobj.NumWorkers == length(range))
	        delete(poolobj);
	        parpool(length(range));
	    end
	end
	nworker = length(range);
else
	nworker = 0;
end
if exist('seed','var')
    rng(seed,gen);
else
    scurr = rng('shuffle')
    seed = scurr.Seed;
end

if exist(ENfilename0, 'dir') && new
    rmdir(ENfilename0,'s');
end
if ~exist(ENfilename0, 'dir')
    mkdir(ENfilename0);
end
copyfile([mfilename,'.m'],[ENfilename0,'/',ENfilename0,'_p.m']);

parfor (i = 1:length(range), nworker)
%for i = 1 
    % for non_cortical_shape edge boundaries
    test_dw = 1;
    test_dh = 1;
    % Objective function weights

    alpha = 1;		% Fitness term weight
	beta = betaRange(i);
	%beta = 100;
    % Training parameters
    % - VFx: Nx points in [0,1], with interpoint separation dx.
	Nx = 14;
	%Nx = range(i)
    rx = [0 1]*nvfx*xRange(i)*nvfRange(i);			% Range of VFx
    % - VFy: ditto for Ny, dy.
	%
	% even
	Ny = 14;
    ry = [0 1]*nvfy*yRange(i)*nvfRange(i);			% Range of VFy
    % - OD: NOD values in range rOD, with interpoint separation dOD.
	l = lRange(i);
	%l = lRange(i);
	%l = l0;
    NOD = 2;			% Number of points along OD
    rOD = [-l l];			% Range of OD
    %  coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
    %  r = 6*l/pi
	%r = range(i)*l;			% OR modulus -- the radius of pinwheel
	%r = r0;
    NOR = 6;	 %8;		% Number of points along OR
	%r = r0*NOR/2*l/pi;
	r = rRange(i);
    % for myCortex patch
    ODnoise = l*0.0;
    ODabsol = 1.0;
    nG = nGrange(i);
	nT = 1;
    %G = round([64 104]*nG);		% Number of centroids *** 
    G = round(G0*nG);		% Number of centroids *** 
	g0 = G;
	G(1) = round(G(1) * aRrange(1,i));
	G(2) = round(G(2) * aRrange(2,i));
	aspectRatio = G(2)/g0(2)/(G(1)/g0(1)); % y/x
    nod = 25;
    a = 0.635; b = 96.7; k = sqrt(140)*0.873145;
    fign = 106;
    ENfilename = [var,'-',num2str(range(i))];
    stats(i) = myV1driver(exchange_nm,seed,ENproc,ENfilename0,ENfilename,non_cortical_LR,cortical_VF,cortical_shape,uniform_LR,test_dw,test_dh,alpha,beta,iters,max_it,Kin,Kend,Nx,rx,Ny,ry,l,NOD,rOD,r,NOR,ODnoise,ODabsol,nG,G,aspectRatio,nT,ecc,nod,a,b,k,i,plots,new,saveLR,separateData,plotting,heteroAlpha,equi,weightType,VFpath,old,randSeed,figlist,rotate);
end
if sum(rRange) > 0 && sum(lRange) > 0
	nnpinw = [stats.npinw];
	nnpinw = nnpinw./mean(nnpinw);
	normD_overMean = [stats.normD_overMean];
	OD_OR_I_ang_overMean = [stats.OD_OR_I_ang_overMean];
	OD_OR_B_ang_overMean = [stats.OD_OR_B_ang_overMean];
	OR_B_ang_overMean = [stats.OR_B_ang_overMean];
	OD_B_ang_overMean = [stats.OD_B_ang_overMean];

	normD_mode = [stats.normD_mode];
	OD_OR_I_angMode = [stats.OD_OR_I_angMode];
	OD_OR_B_angMode = [stats.OD_OR_B_angMode];
	OD_B_angMode = [stats.OD_B_angMode];
	OR_B_angMode = [stats.OR_B_angMode];
	h = figure((length(range)+1)*1000+1);
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
	
	set(h,'PaperPositionMode','auto')
	print(h, figname, format, '-r150');
	saveas(h,[figname,'.fig']);
end

eval(['save ', ENfilename0 '/' ENfilename0 '-stats.mat stats']); 