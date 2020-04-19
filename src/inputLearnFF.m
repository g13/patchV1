% essential input files
fLGN_V1_ID = 'LGN_V1_idList_lFF.bin'
fLGN_V1_s = 'LGN_V1_sList_lFF.bin'
fLGN_surfaceID = 'LGN_surfaceID_lFF.bin'
fV1_pos = 'V1_pos_lFF.bin'
fLGN_vpos = 'LGN_vpos_lFF.bin'

parvoMagno = 1 % parvo
%parvoMagno = 2 % magno 

initialConnectionStrength = 1; % also can be changed by ratioLGN in .cfg file
iCS_std = 0.2;
nblock = 1;
blockSize = 1024;
nV1 = nblock*blockSize;
nLGN_1D = 16;
frameVisV1output = false; % if need framed V1 output, write visual pos to fV1_pos

nLGN = nLGN_1D*nLGN_1D;

%nLGNperV1 = round(nLGN * 0.2)
nLGNperV1 = 200;

fid = fopen(fLGN_V1_ID, 'w'); % format follows read_listOfList in util/util.h
for i = 1:nV1
    fwrite(fid, nLGNperV1, 'uint');
    id = randperm(nLGN, nLGNperV1)-1; % index start from 0
    fwrite(fid, id, 'uint');
end
fclose(fid);


sLGN = randn(nLGNperV1,nV1)*iCS_std+initialConnectionStrength;
fid = fopen(fLGN_V1_s, 'w'); % format follows function read_LGN in patch.h
fwrite(fid, nV1, 'uint');
fwrite(fid, nLGNperV1, 'uint');
for i = 1:nV1
    fwrite(fid, nLGNperV1, 'uint');
    fwrite(fid, sLGN(:, i), 'float');
end
fclose(fid);


fid = fopen(fLGN_surfaceID, 'w'); % format follows patch.cu, search for the same variable name
[LGN_idx, LGN_idy] = meshgrid(1:nLGN_1D, 1:nLGN_1D); % meshgrid gives coordinates of LGN on the surface memory
% tranform the coords into row-first order
LGN_idx = LGN_idx'-1; % index start with 0 in c/c++
LGN_idy = LGN_idy'-1;
fwrite(fid, nLGN_1D-1, 'uint'); % width - 1, legacy problem, will add 1 after read
fwrite(fid, nLGN_1D-1, 'uint'); % height - 1
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
LGN_x = (LGN_idx(:)-nLGN_1D/2+0.5)./nLGN_1D.*max_ecc./0.5;
LGN_y = (LGN_idy(:)-nLGN_1D/2+0.5)./nLGN_1D.*max_ecc./0.5;
disp([min(LGN_x), max(LGN_x)]);
disp([min(LGN_y), max(LGN_y)]);
fwrite(fid, LGN_x, 'float');
fwrite(fid, LGN_y, 'float');
% see preprocess/RFtype.h
if parvoMagno == 1
    LGN_type = randi(4,[nLGN,1])-1; % 0-3: L-on, L-off, M-on, M-off
end
if parvoMagno == 2
    LGN_type = 3+randi(2,[nLGN,1]); % 4:on center or 5: off center
end
fwrite(fid, LGN_type, 'uint');
% in polar form
LGN_ecc = sqrt(LGN_x.*LGN_x + LGN_y.*LGN_y); %degree
LGN_polar = atan2(LGN_y, LGN_x); % [-pi, pi]
fwrite(fid, LGN_polar, 'float');
fwrite(fid, LGN_ecc, 'float');
fclose(fid);

% by setting useNewLGN to false, populate LGN.bin follow the format in patch.cu

% not in use, only self block is included
fNeighborBlock = 'neighborBlock_lFF.bin'
% not in use, may use if cortical inhibition is needed
fV1_delayMat = 'V1_delayMat_lFF.bin' % zeros
fV1_conMat = 'V1_conMat_lFF.bin' % zeros
fV1_vec = 'V1_vec_lFF.bin' % zeros

nearNeighborBlock = 1; % self-only
fid = fopen(fV1_conMat, 'w');
fwrite(fid, nearNeighborBlock , 'uint');
            %(post, pre)
conMat = zeros(nV1,nV1);
fwrite(fid, conMat , 'uint');
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
fwrite(fid, delayMat , 'uint');
fclose(fid);

fid = fopen(fNeighborBlock, 'w');
nNeighborBlock = [1];
fwrite(fid, nNeighborBlock, 'uint'); % number of neighbor
nBlockId = [[0]]
fwrite(fid, nBlockId, 'uint'); % number of neighbor
fclose(fid);


fid = fopen(fV1_vec, 'w');
nVec = zeros(nV1,1);
fwrite(fid, nVec, 'uint');
fclose(fid);
