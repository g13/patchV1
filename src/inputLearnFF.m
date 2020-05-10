% essential input files
suffix = 'lFF'
if ~isempty(suffix)
    suffix = ['_', suffix];
end
fLGN_V1_ID = ['LGN_V1_idList', suffix, '.bin'];
fLGN_V1_s = ['LGN_V1_sList', suffix, '.bin'];
fLGN_surfaceID = ['LGN_surfaceID', suffix, '.bin'];
fV1_pos = ['V1_pos', suffix, '.bin'];
fLGN_vpos = ['LGN_vpos', suffix, '.bin'];
fLGN_switch = ['LGN_switch', suffix, '.bin'];

%parvoMagno = 1 % parvo
parvoMagno = 2 % magno 
%parvoMagno = 3 % both, don't use, only for testing purpose

normed = true;
initialConnectionStrength = 0.01; % also can be changed by ratioLGN in .cfg file
eiStrength = 0.000;
ieStrength = 0.000;
iCS_std = 1.0; % zero std gives single-valued connection strength
nblock = 1;
blockSize = 1024;
mE = 768;
mI = 256;
nV1 = nblock*blockSize;
nLGN_1D = 15;
double = false;
frameVisV1output = false; % if need framed V1 output, write visual pos to fV1_pos

nStatus = 2;
status = zeros(6,nStatus);
status(5,1) = 1;
status(6,1) = 0;
status(5,2) = 0;
status(6,2) = 1;
%statusDur = [1];
statusDur = [1/2; 1/2];

nLGN = nLGN_1D*nLGN_1D;
if double
    nLGN = nLGN * 2;
end

nLGNperV1 = round(nLGN * 0.9)
nI = int32(nLGNperV1/4);
nE = int32(nLGNperV1);

fid = fopen(fLGN_V1_ID, 'w'); % format follows read_listOfList in util/util.h
id = zeros(nLGNperV1,nV1);
for i = 1:nV1
    fwrite(fid, nLGNperV1, 'uint');
    id(:,i) = randperm(nLGN, nLGNperV1)-1; % index start from 0
    fwrite(fid, id(:,i), 'uint');
end
fclose(fid);


fid = fopen(fLGN_surfaceID, 'w'); % format follows patch.cu, search for the same variable name
if double 
    [LGN_idx, LGN_idy] = meshgrid(1:2:nLGN_1D*2, 1:2:nLGN_1D*2); % meshgrid gives coordinates of LGN on the surface memory
    [LGN_idx2, LGN_idy2] = meshgrid(2:2:nLGN_1D*2, 2:2:nLGN_1D*2); % meshgrid gives coordinates of LGN on the surface memory
    LGN_idx = [LGN_idx'-1, LGN_idx2'-1];
    LGN_idy = [LGN_idy'-1, LGN_idy2'-1];
else
    [LGN_idx, LGN_idy] = meshgrid(1:nLGN_1D, 1:nLGN_1D); % meshgrid gives coordinates of LGN on the surface memory
    % tranform the coords into row-first order
    LGN_idx = LGN_idx'-1; % index start with 0 in c/c++
    LGN_idy = LGN_idy'-1;
end
if double 
    fwrite(fid, 2*nLGN_1D-1, 'uint'); % width - 1, legacy problem, will add 1 after read
    fwrite(fid, 2*nLGN_1D-1, 'uint'); % height - 1
else
    fwrite(fid, nLGN_1D-1, 'uint'); % width - 1, legacy problem, will add 1 after read
    fwrite(fid, nLGN_1D-1, 'uint'); % height - 1
end
fwrite(fid, LGN_idx(:), 'int');
fwrite(fid, LGN_idy(:), 'int');
fclose(fid);

fid = fopen(fV1_pos, 'w'); % format follows patch.cu, search for the same variable name
% not necessarily a ring
loop = linspace(0,pi,nV1+1);
polar = loop(1:nV1);
V1_pos = [cos(polar)', sin(polar)'];
fwrite(fid, nblock, 'uint');
fwrite(fid, blockSize, 'uint');
fwrite(fid, 2, 'uint'); %posDim, don't change
%
fwrite(fid, -1, 'double'); % x0: min(x) = cos(pi);
fwrite(fid, 2, 'double'); % xspan: max(x) - min(x) = cos(0)-cos(pi);
fwrite(fid, -1, 'double'); % y0..
fwrite(fid, 2, 'double'); % yspan..
fwrite(fid, V1_pos, 'double'); % write V1_pos
if frameVisV1output % change if true, default to the same as physical position
    fwrite(fid, -1, 'double'); % vx0: min(vx)
    fwrite(fid, 2, 'double'); % vxspan: max(vx)-min(vx)
    fwrite(fid, -1, 'double'); % vy..
    fwrite(fid, 2, 'double'); % vyspan..
    fwrite(fid, V1_pos, 'double'); 
end
fclose(fid);

fid = fopen(fLGN_switch, 'w');
fwrite(fid, nStatus, 'uint'); 
fwrite(fid, status, 'float'); % activated percentage
fwrite(fid, statusDur, 'float'); % duration
fclose(fid);

fid = fopen(fLGN_vpos, 'w'); % format follows patch.cu, search for the same variable name
fwrite(fid, nLGN, 'uint'); % # ipsi-lateral LGN 
fwrite(fid, 0, 'uint'); % contra-lateral LGN all from one eye
max_ecc = 0.25;
fwrite(fid, max_ecc, 'float');
% using uniform_retina, thus normalized to the central vision of a single eye
LGN_pos = zeros(nLGN,2);
fwrite(fid, -max_ecc, 'float'); % x0
fwrite(fid, 2*max_ecc, 'float'); % xspan
fwrite(fid, -max_ecc, 'float'); % y0
fwrite(fid, 2*max_ecc, 'float'); % yspan
if double 
    LGN_idx = [LGN_idx2', LGN_idx2']/2-1;
    LGN_idy = [LGN_idy2', LGN_idy2']/2-1;
    LGN_x = (LGN_idx(:)-nLGN_1D/2+0.5)./nLGN_1D.*max_ecc./0.5;
    LGN_y = (LGN_idy(:)-nLGN_1D/2+0.5)./nLGN_1D.*max_ecc./0.5;
else
    LGN_x = (LGN_idx(:)-nLGN_1D/2+0.5)./nLGN_1D.*max_ecc./0.5;
    LGN_y = (LGN_idy(:)-nLGN_1D/2+0.5)./nLGN_1D.*max_ecc./0.5;
end
disp([min(LGN_x), max(LGN_x)]);
disp([min(LGN_y), max(LGN_y)]);
fwrite(fid, LGN_x, 'float');
fwrite(fid, LGN_y, 'float');

% see preprocess/RFtype.h
if parvoMagno == 1
    LGN_type = randi(4,[nLGN,1])-1; % 0-3: L-on, L-off, M-on, M-off
end
if parvoMagno == 2
    %LGN_type = 3+randi(2,[nLGN,1]); % 4:on center or 5: off center
    %LGN_type = 4+zeros(nLGN,1); % 4:on center or 5: off center
    if double
        LGN_type = zeros(nLGN_1D*nLGN_1D, 2);
        LGN_type(:,1) = 4;
        LGN_type(:,2) = 5;
    else
        type0 = zeros(nLGN_1D, 1);
        type0(1:2:nLGN_1D) = 4;
        type0(2:2:nLGN_1D) = 5;
        type1 = zeros(nLGN_1D, 1);
        type1(1:2:nLGN_1D) = 5;
        type1(2:2:nLGN_1D) = 4;
        for i = 1:nLGN_1D
            if mod(i,2) == 1
                %LGN_type(:,i) = circshift(type, i);
                LGN_type(:,i) = type0;
            else
                LGN_type(:,i) = type1;
            end
        end
    end
end
if parvoMagno == 3
    LGN_type = zeros(nLGN,1);
    nParvo = randi(nLGN, 1)
    nMagno = nLGN - nParvo
    LGN_type(1:nParvo) = randi(4,[nParvo,1])-1; %
    LGN_type((nParvo+1):nLGN) = 3 + randi(2,[nMagno,1]); %
end
fwrite(fid, LGN_type, 'uint');
% in polar form
LGN_ecc = sqrt(LGN_x.*LGN_x + LGN_y.*LGN_y); %degree
LGN_polar = atan2(LGN_y, LGN_x); % [-pi, pi]
fwrite(fid, LGN_polar, 'float');
fwrite(fid, LGN_ecc, 'float');
fclose(fid);

if normed
    sLGN = zeros(nLGNperV1, nV1);
    for i = 1:nV1
        ss = exp(-(LGN_ecc(id(:,i)+1) ./ (max_ecc*iCS_std)).^2);
        sLGN(:,i) = ss./sum(ss)*initialConnectionStrength*nLGNperV1;
    end
else
    sLGN = randn(nLGNperV1,nV1)*iCS_std+initialConnectionStrength;
end
fid = fopen(fLGN_V1_s, 'w'); % format follows function read_LGN in patch.h
fwrite(fid, nV1, 'uint');
fwrite(fid, nLGNperV1, 'uint');
for i = 1:nV1
    fwrite(fid, nLGNperV1, 'uint');
    fwrite(fid, sLGN(:, i), 'float');
end
fclose(fid);

% by setting useNewLGN to false, populate LGN.bin follow the format in patch.cu

% not in use, only self block is included
fNeighborBlock = ['neighborBlock', suffix, '.bin'];
% not in use, may use if cortical inhibition is needed
fV1_delayMat = ['V1_delayMat', suffix, '.bin']; % zeros
fV1_conMat = ['V1_conMat', suffix, '.bin']; % zeros
fV1_vec = ['V1_vec', suffix, '.bin']; % zeros

nearNeighborBlock = 1; % self-only
fid = fopen(fV1_conMat, 'w');
fwrite(fid, nearNeighborBlock , 'uint');
            %(post, pre)
conMat = zeros(nV1,nV1);
conIE = zeros(mI,mE);
for i = 1:mI
    ied = randperm(mE, nE);
    conIE(i,ied) = ieStrength;
end
conEI = zeros(mE,mI);
for i = 1:mE
    eid = randperm(mI, nI);
    conEI(i,eid) = eiStrength;
end
conMat(1:mE,(mE+1):blockSize) = conEI;
conMat((mE+1):blockSize,1:mE) = conIE;
fwrite(fid, conMat , 'float');
fclose(fid);

fid = fopen(fV1_delayMat, 'w');
fwrite(fid, nearNeighborBlock , 'uint');
delayMat = zeros(nV1);
% not necessary to calc distance
for i=1:nV1
    for j=1:nV1
        delayMat(i,j) = norm(V1_pos(i,:) - V1_pos(j,:), 2);
        delayMat(j,i) = delayMat(i,j);
    end
end
%
fwrite(fid, delayMat , 'float');
fclose(fid);

fid = fopen(fNeighborBlock, 'w');
nNeighborBlock = [1];
fwrite(fid, nNeighborBlock, 'uint'); % number of neighbor
nBlockId = [[0]]
fwrite(fid, nBlockId, 'uint'); % id of neighbor
fclose(fid);


fid = fopen(fV1_vec, 'w');
nVec = zeros(nV1,1);
fwrite(fid, nVec, 'uint');
fclose(fid);
