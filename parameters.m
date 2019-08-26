clear all
clc;
 
gen = 'twister';
clear seed
seed = 1425269650;
%seed = 1422230161;
format = '-dpng';

plots = true;
saveLR = true;
new = true;
ENproc = 'var';		%2One of 'var', 'save', 'varplot', 'saveplot'
% Processing of intermediate (historical) parameters:
var = 'nG';
ENfilename0 = ['cortex_94-',var];   % Simulation name ***
range = [1,2,4,6,8];
non_cortical_lr = false;
% non_cortical_lr = true;
% cortical_shape = false;
cortical_shape = true;
uniform_LR = true;
% uniform_LR = false;

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
<<<<<<< HEAD
for i = 1:length(range)
=======
%for i = 1:length(range)
	ratio = range(i)/range(end);
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
    % for non_cortical_shape edge boundaries
    test_dw = 5;
    test_dh = 7;
    % Objective function weights

    alpha = 1;		% Fitness term weight
	%beta = 200*range(i)/range(end);
<<<<<<< HEAD
	beta = 100;
=======
	beta = 150;
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
    % Training parameters
    iters = 10;%21;			% No. of annealing iterations (saved)
    max_it = 10;%12;			% No. of annealing iterations (not saved) ***
    Kin = 0.15;			% Initial K ***
    Kend = 0.03;        % Final K ***    
    % - VFx: Nx points in [0,1], with interpoint separation dx.
    Nx = 8;			% Number of points along VFx ***
<<<<<<< HEAD
	nvf = 8;
=======
	nvf = 10*ratio;
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
    %nvf = 6 + range(i);
    rx = [0 0.16]*nvf;			% Range of VFx
    dx = diff(rx)/(Nx-1);		% Separation between points along VFx
    % - VFy: ditto for Ny, dy.
<<<<<<< HEAD
	Ny = 12;			% Number of points along VFy ***
=======
	Ny = 13;			% Number of points along VFy ***
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
    ry = [0 0.24]*nvf;			% Range of VFy
    dy = diff(ry)/(Ny-1);		% Separation between points along VFy    
    d = (dx + dy)/2;
    % - OD: NOD values in range rOD, with interpoint separation dOD.
	%l = 0.08 + 0.1*range(i)/range(end);			% Half-separation between OD layers ***
	l = 0.12*ratio;
    NOD = 2;			% Number of points along OD
    rOD = [-l l];			% Range of OD
    dOD = diff(rOD)/(NOD-1);	% Separation between points along OD
    %  coded as NOR Cartesian-coordinate pairs (ORx,ORy). -- later by pol2cart
    %  r = 6*l/pi
	%r = (0.8+range(i)/range(end))*l;			% OR modulus -- the radius of pinwheel
<<<<<<< HEAD
	r = 1.2*l;
=======
	r = 1.4*l;
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
    NOR = 8;	 %8;		% Number of points along OR    
    % for myCortex patch
    ODnoise = l*0.0;
    ODabsol = 1.0;
<<<<<<< HEAD
    nG = range(i);
=======
    nG = 4;
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
    G = round([64 104]*nG);		% Number of centroids *** 
    ecc = 2;
    nod = 25;
    a = 0.635; b = 96.7; k = sqrt(140)*0.873145;
    fign = 106;
    ENfilename = [var,'-',num2str(range(i))];
<<<<<<< HEAD
	if i == 1
		saveLR = true;
	else
		saveLR = false;
	end
=======
>>>>>>> f981afccf594495551cf7b1d9065b8a34ed724bb
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
